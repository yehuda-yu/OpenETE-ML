import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.feature_selection import SelectKBest, f_regression, RFE
import plotly.graph_objects as go
import plotly.subplots as sp
import plotly.express as px
import itertools
from scipy import stats
from scipy.ndimage.filters import maximum_filter, minimum_filter
from sklearn.linear_model import LassoCV, LassoLarsCV, LarsCV, Lasso, OrthogonalMatchingPursuitCV, LassoLars, OrthogonalMatchingPursuit, ElasticNetCV, ElasticNet, TweedieRegressor, HuberRegressor, RANSACRegressor, LinearRegression, BayesianRidge, Ridge, LassoLarsIC, Lars, PassiveAggressiveRegressor, SGDRegressor, RidgeCV
from sklearn.svm import SVR, NuSVR
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import ExtraTreesRegressor, AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor, BaggingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.kernel_ridge import KernelRidge
import lightgbm as lgb
import xgboost as xgb
from scipy.stats import uniform, randint
from scipy.stats import loguniform
#import lazypredict
#from lazypredict.Supervised import LazyRegressor
import LazyRegressor
from functools import wraps



def optimized_cache(func):
    @wraps(func)
    @st.cache_data
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

@optimized_cache
def perform_pca(data, target_column, categorical_columns, variance_percentage):
    """
    Performs Principal Component Analysis (PCA) on numerical columns of a DataFrame,
    retaining components that explain a specified percentage of variance.

    Args:
        data (pd.DataFrame): The input DataFrame containing features and target.
        target_column (str): The name of the target column.
        categorical_columns (list): A list of categorical column names.
        variance_percentage (float): The desired percentage of variance to explain.

    Returns:
        tuple: A tuple containing:
            - df_final (pd.DataFrame): The DataFrame with reduced dimensions after PCA.
            - total_cols_before (int): The total number of columns before PCA.
            - total_cols_after (int): The total number of columns after PCA.
            - cum_var (np.ndarray): The cumulative explained variance ratios.
    """
    # Separate features and target
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Identify numerical columns (excluding the target and categorical columns)
    numerical_columns = [col for col in X.columns if col not in categorical_columns]

    # Standardize the numerical columns
    X[numerical_columns] = StandardScaler().fit_transform(X[numerical_columns])

    # Perform PCA
    pca = PCA()
    X_pca = pca.fit_transform(X[numerical_columns])

    # Create DataFrame with principal components
    n_components = pca.n_components_
    pc_col_names = [f"PC_{i+1}" for i in range(n_components)]
    df_pca = pd.DataFrame(data=X_pca, columns=pc_col_names)

    # Calculate the cumulative percentage of explained variance
    cum_var = np.cumsum(pca.explained_variance_ratio_)

    # Determine the number of components needed to explain variance_percentage% of the variance
    n_components_to_keep = np.argmax(cum_var >= variance_percentage / 100) + 1

    # Keep only the selected principal components
    df_pca_reduced = df_pca.iloc[:, :n_components_to_keep]

    # Add back the target column and categorical columns
    df_final = pd.concat([df_pca_reduced, data[categorical_columns], y], axis=1)

    # Calculate the total number of columns before and after PCA
    total_cols_before = X.shape[1] + len(categorical_columns)
    total_cols_after = df_final.shape[1] - 1  # Exclude the target column

    return df_final, total_cols_before, total_cols_after, cum_var

@optimized_cache
def plot_cumulative_variance(cum_var, variance_percentage):
    """
    Plot the cumulative explained variance ratio.
    
    Args:
    cum_var (np.array): Cumulative explained variance ratio
    variance_percentage (float): Desired percentage of variance to explain
    """
    # Create trace for cumulative variance
    trace = go.Scatter(
        x=list(range(1, len(cum_var) + 1)),
        y=cum_var * 100,
        mode='lines+markers',
        name='Cumulative Explained Variance',
        hoverinfo='x+y',
        line=dict(color='royalblue')
    )

    # Create trace for variance threshold
    threshold_trace = go.Scatter(
        x=[1, len(cum_var)],
        y=[variance_percentage, variance_percentage],
        mode='lines',
        name=f'{variance_percentage}% Variance Threshold',
        line=dict(color='red', dash='dash')
    )

    # Calculate number of components needed to explain desired variance
    n_components = np.argmax(cum_var >= variance_percentage / 100) + 1

    # Create layout
    layout = go.Layout(
        title='Cumulative Explained Variance Ratio',
        xaxis=dict(title='Number of Components'),
        yaxis=dict(title='Cumulative Explained Variance (%)', range=[0, 100]),
        showlegend=True,
        annotations=[
            dict(
                x=n_components,
                y=variance_percentage,
                xref='x',
                yref='y',
                text=f'{n_components} components<br>explain {variance_percentage}% variance',
                showarrow=True,
                arrowhead=7,
                ax=0,
                ay=-40
            )
        ]
    )

    # Create figure and plot
    fig = go.Figure(data=[trace, threshold_trace], layout=layout)
    st.plotly_chart(fig, use_container_width=True)

    # Display additional information
    st.write(f"Number of components needed to explain {variance_percentage}% of variance: {n_components}")

@optimized_cache
def perform_tsne(data, target_column, categorical_columns, n_components=2):
    """
    Performs t-SNE on the numerical columns of a DataFrame.
    
    Args:
        data (pd.DataFrame): The input DataFrame containing features and target.
        target_column (str): The name of the target column.
        categorical_columns (list): A list of categorical column names.
        n_components (int): The number of dimensions to reduce to (default is 2).
    
    Returns:
        pd.DataFrame: The DataFrame with the t-SNE transformed features.
    """
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    numerical_columns = [col for col in X.columns if col not in categorical_columns]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X[numerical_columns])
    
    tsne = TSNE(n_components=n_components, random_state=42)
    X_tsne = tsne.fit_transform(X_scaled)
    
    df_tsne = pd.DataFrame(X_tsne, columns=[f"tSNE_{i+1}" for i in range(n_components)])
    df_final = pd.concat([df_tsne, data[categorical_columns], y], axis=1)
    
    if n_components == 2:
        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(df_tsne['tSNE_1'], df_tsne['tSNE_2'], c=y, cmap='viridis', alpha=0.7)
        plt.colorbar(scatter)
        plt.title('t-SNE Visualization (2D)')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        st.pyplot(fig)
    elif n_components == 3:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(df_tsne['tSNE_1'], df_tsne['tSNE_2'], df_tsne['tSNE_3'], c=y, cmap='viridis', alpha=0.7)
        plt.colorbar(scatter)
        ax.set_title('t-SNE Visualization (3D)')
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        ax.set_zlabel('t-SNE 3')
        st.pyplot(fig)
    else:
        st.write("Visualization is only available for 2 or 3 components. For higher dimensions, please refer to the data table.")
    
    st.write(f"Number of features before t-SNE: {len(numerical_columns)}")
    st.write(f"Number of features after t-SNE: {n_components}")
    
    return df_final

@optimized_cache
def perform_umap(data, target_column, categorical_columns, n_components=2, n_neighbors=15, min_dist=0.1, metric='euclidean'):
    """
    Performs UMAP (Uniform Manifold Approximation and Projection) on the numerical columns of a DataFrame.
    
    Args:
        data (pd.DataFrame): The input DataFrame containing features and target.
        target_column (str): The name of the target column.
        categorical_columns (list): A list of categorical column names.
        n_components (int): The number of dimensions to reduce to (default is 2).
        n_neighbors (int): The size of local neighborhood for manifold approximation (default is 15).
        min_dist (float): The minimum distance between points in low dimensional representation (default is 0.1).
        metric (str): The metric to use for distance computation (default is 'euclidean').
    
    Returns:
        pd.DataFrame: The DataFrame with the UMAP transformed features.
    """
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    numerical_columns = [col for col in X.columns if col not in categorical_columns]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X[numerical_columns])
    
    umap_model = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, 
                           min_dist=min_dist, metric=metric, random_state=42)
    X_umap = umap_model.fit_transform(X_scaled)
    
    df_umap = pd.DataFrame(X_umap, columns=[f"UMAP_{i+1}" for i in range(n_components)])
    df_final = pd.concat([df_umap, data[categorical_columns], y], axis=1)
    
    if n_components == 2:
        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(df_umap['UMAP_1'], df_umap['UMAP_2'], c=y, cmap='viridis', alpha=0.7)
        plt.colorbar(scatter)
        plt.title('UMAP Visualization (2D)')
        plt.xlabel('UMAP 1')
        plt.ylabel('UMAP 2')
        st.pyplot(fig)
    elif n_components == 3:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(df_umap['UMAP_1'], df_umap['UMAP_2'], df_umap['UMAP_3'], c=y, cmap='viridis', alpha=0.7)
        plt.colorbar(scatter)
        ax.set_title('UMAP Visualization (3D)')
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.set_zlabel('UMAP 3')
        st.pyplot(fig)
    else:
        st.write("Visualization is only available for 2 or 3 components. For higher dimensions, please refer to the data table.")
    
    st.write(f"Number of features before UMAP: {len(numerical_columns)}")
    st.write(f"Number of features after UMAP: {n_components}")
    
    return df_final

def perform_rfe(data, target_column, categorical_columns, n_features_to_select):
    """
    Performs Recursive Feature Elimination (RFE) on the numerical columns of a DataFrame.
    
    Args:
        data (pd.DataFrame): The input DataFrame containing features and target.
        target_column (str): The name of the target column.
        categorical_columns (list): A list of categorical column names.
        n_features_to_select (int): The number of features to select.
    
    Returns:
        pd.DataFrame: The DataFrame with the selected features.
    """
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    # Identify numerical columns
    numerical_columns = [col for col in X.columns if col not in categorical_columns]
    
    # Standardize the numerical columns
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X[numerical_columns]), columns=numerical_columns)
    
    # Add back categorical columns
    X_scaled = pd.concat([X_scaled, X[categorical_columns]], axis=1)
    
    # Perform RFE
    estimator = RandomForestRegressor(n_estimators=100, random_state=42)
    selector = RFE(estimator, n_features_to_select=n_features_to_select, step=1)
    selector = selector.fit(X_scaled, y)
    
    # Get selected feature names
    selected_features = X_scaled.columns[selector.support_].tolist()
    
    # Create DataFrame with selected features and target
    df_selected = pd.concat([data[selected_features], y], axis=1)
    
    return df_selected

@optimized_cache
def time_series_feature_extraction(data, target_col, categorical_columns):
    """
    Extract time series features from a DataFrame.

    Parameters
    ----------
    data : pandas DataFrame
        The input DataFrame with hyperspectral data.
    target_col : str
        The name of the target column in the DataFrame.
    categorical_columns : list 
        A list of column names that are categorical and you want to keep in the DataFrame.

    Returns
    -------
    pandas DataFrame
        The output DataFrame with computed features and the target column.
    """
    # Save the target data 
    target_col_data = data[target_col].values
    categorical_columns_data = data[categorical_columns]
    
    data = data.drop(columns=categorical_columns + [target_col]).select_dtypes(include=np.number)
    
    features = {
        'Mean': data.mean(axis=1),
        'Median': data.median(axis=1),
        'Std': data.std(axis=1),
        'Percent_Beyond_Std': (np.abs(data - data.mean(axis=1)[:, None]) > data.std(axis=1)[:, None]).sum(axis=1) / data.shape[1],
        'Amplitude': data.max(axis=1) - data.min(axis=1),
        'Max': data.max(axis=1),
        'Min': data.min(axis=1),
        'Max_Slope': np.abs(np.diff(data, axis=1)).max(axis=1),
        'MAD': np.median(np.abs(data - np.median(data, axis=1)[:, None]), axis=1),
        'Percent_Close_to_Median': (np.abs(data - np.median(data, axis=1)[:, None]) < 0.5 * np.median(data, axis=1)[:, None]).sum(axis=1) / data.shape[1] * 100,
        'Skew': data.skew(axis=1),
        'Flux_Percentile': np.percentile(data, 90, axis=1),
        'Weighted_Average': np.average(data, weights=data.columns.astype(float), axis=1)
    }
    
    parameters_df = pd.DataFrame(features)
    parameters_df['Percent_Difference_Flux_Percentile'] = parameters_df['Flux_Percentile'].pct_change().fillna(0)
    
    return pd.concat([parameters_df, categorical_columns_data, pd.Series(target_col_data, name=target_col)], axis=1)

def NDSI_pearson(data,categorical_columns,  target_col):
    '''
    Calculates the Pearson correlation coefficient and p-value
    between the normalized difference spectral index (NDSI) and the target column.

    Parameters:
    - data: DataFrame, input data containing spectral bands and target column
    - categorical_columns: list, list of the categorial colums to drop before ndsi
    - target_col: str, name of the target column

    Returns:
    - df_results: DataFrame, contains band pairs, Pearson correlation, p-value, and absolute Pearson correlation
    '''

    # Extract labels column
    y = data[target_col].values
    # Delete target column from features dataframe
    df = data.drop(target_col, axis=1)
    # drop non numeric columns
    df = df.drop(categorical_columns, axis=1)
    # Convert column names to str
    df.columns = df.columns.map(str)
    bands_list = df.columns

    # All possible pairs of columns
    all_pairs = list(itertools.combinations(bands_list, 2))

    # Initialize arrays for correlation values and p-values
    corrs = np.zeros(len(all_pairs))
    pvals = np.zeros(len(all_pairs))

    # Calculate the NDSI and Pearson correlation
    progress_bar = st.progress(0)
    for index, pair in enumerate(all_pairs):
        a = df[pair[0]].values
        b = df[pair[1]].values
        Norm_index = (a - b) / (a + b)
        # Pearson correlation and p-value
        corr, pval = stats.pearsonr(Norm_index, y)
        corrs[index] = corr
        pvals[index] = pval
        # Update progress bar
        progress_bar.progress((index + 1) / len(all_pairs))

    # Convert results to DataFrame
    col1 = [tple[0] for tple in all_pairs]
    col2 = [tple[1] for tple in all_pairs]
    index_col = [f"{tple[0]},{tple[1]}" for tple in all_pairs]
    data = {'band1': col1, "band2": col2, 'Pearson_Corr': corrs, 'p_value': pvals}
    df_results = pd.DataFrame(data=data, index=index_col)
    df_results["Abs_Pearson_Corr"] = df_results["Pearson_Corr"].abs()
    
    return df_results.sort_values('Abs_Pearson_Corr', ascending=False)

@optimized_cache
def display_ndsi_heatmap(results, threshold, max_distance):
    """
    Display a heatmap with local minima and maxima points based on Pearson correlation values.

    Parameters:
    - results : DataFrame
        DataFrame containing Pearson correlation values between spectral bands.
    - threshold : float
        Threshold value for identifying local minima and maxima points.
    - max_distance : int
        Maximum distance for local minima and maxima identification.

    Returns:
    - top_bands_list : list of tuples
        List of tuples where each tuple contains the names of two spectral bands
        corresponding to the local minima and maxima points.
    """
    # Pivot the dataframe to have bands as rows and columns
    data = results.pivot(index='band1', columns='band2', values='Pearson_Corr')

    # Find local maxima and minima exceeding the threshold
    local_max = (maximum_filter(data, footprint=np.ones((max_distance, max_distance))) == data) & (data > threshold)
    local_min = (minimum_filter(data, footprint=np.ones((max_distance, max_distance))) == data) & (data < -threshold)

    # Get indices of local maxima and minima
    maxima_x, maxima_y = np.where(local_max)
    minima_x, minima_y = np.where(local_min)

    # Create lists to store band1 and band2 indices for minima and maxima
    minima_list = [(data.index[minima_x][i], data.columns[minima_y][i]) for i in range(len(minima_x))]
    maxima_list = [(data.index[maxima_x][i], data.columns[maxima_y][i]) for i in range(len(maxima_x))]

    # Merge the two lists into one list
    top_bands_list = minima_list + maxima_list

    # Create a Plotly heatmap
    fig = go.Figure(data=go.Heatmap(
        z=data.values,
        x=data.columns,
        y=data.index,
        colorscale='RdBu',  # Choose the color scale
        zmin=-1, zmax=1,  # Set the color scale range
        colorbar=dict(title='Pearson Correlation')  # Add colorbar title
    ))
    
    # Add local maxima and minima (without legend)
    maxima_x, maxima_y = np.where(local_max)
    fig.add_trace(go.Scatter(
        x=data.columns[maxima_y], 
        y=data.index[maxima_x], 
        mode='markers', 
        marker=dict(color='blue', size=8), 
        name='Local Maxima',
        showlegend=False,  # Hide from legend
        hovertemplate='<b>Local Maximum</b><br>Band 1: %{y}<br>Band 2: %{x}<br><extra></extra>'
    ))

    minima_x, minima_y = np.where(local_min)
    fig.add_trace(go.Scatter(
        x=data.columns[minima_y], 
        y=data.index[minima_x], 
        mode='markers', 
        marker=dict(color='red', size=8), 
        name='Local Minima',
        showlegend=False,  # Hide from legend
        hovertemplate='<b>Local Minimum</b><br>Band 1: %{y}<br>Band 2: %{x}<br><extra></extra>'
    ))

    # Update layout (hide legend entirely)
    fig.update_layout(
        title='NDSI Heatmap - Hover for Values',
        xaxis_title='Band 2',
        yaxis_title='Band 1',
        height=600,  # Adjust height as needed
        width=800,  # Adjust width as needed
        template='plotly',  # Choose plotly theme
        showlegend=False  # Hide legend completely
    )

    # Display the Plotly figure using st.plotly_chart()
    st.plotly_chart(fig)

    return top_bands_list

@optimized_cache
def calculate_ndsi(data, top_bands_list):
    """
    Calculate Normalized Difference Spectral Index (NDSI) for each pair of spectral bands.

    Parameters:
    - data : DataFrame
        Original data containing spectral bands.
    - top_bands_list : list of tuples
        List where each tuple contains the names of two spectral bands.

    Returns:
    - ndsi_df : DataFrame
        DataFrame where each column represents a pair of spectral bands,
        and the values are the corresponding NDSI values calculated as (a - b) / (a + b),
        where 'a' and 'b' are the values of the respective spectral bands.
    """
    ndsi_df = pd.DataFrame()

    # Calculate NDSI for each tuple in the list
    for tup in top_bands_list:
        a = data[tup[0]]
        b = data[tup[1]]
        ndsi = (a - b) / (a + b)
        column_name = f"{tup[0]}-{tup[1]}"
        ndsi_df[column_name] = ndsi

    return ndsi_df

def perform_select_kbest(data, target_column, categorical_columns, k):
    """
    Performs SelectKBest feature selection on the DataFrame.

    Args:
        data (pd.DataFrame): The input DataFrame containing features and target.
        target_column (str): The name of the target column.
        categorical_columns (list): A list of categorical column names.
        k (int): The number of top features to select.

    Returns:
        pd.DataFrame: The DataFrame with the selected features.
    """
    # Separate features and target
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Identify numerical columns (excluding the target and categorical columns)
    numerical_columns = [col for col in X.columns if col not in categorical_columns]

    # Perform SelectKBest
    selector = SelectKBest(f_regression, k=k)
    X_new = selector.fit_transform(X[numerical_columns], y)

    # Create a DataFrame with the selected features
    df_selected = pd.DataFrame(X_new, columns=[numerical_columns[i] for i in selector.get_support(indices=True)])

    # Add back the target column and categorical columns
    df_final = pd.concat([df_selected, data[categorical_columns], y], axis=1)

    return df_final

def perform_boruta(data, target_column, categorical_columns, max_iter=100, alpha=0.05):
    """
    Performs Boruta all-relevant feature selection on the DataFrame.
    Boruta finds ALL features that are relevant for prediction, not just a minimal set.

    Args:
        data (pd.DataFrame): The input DataFrame containing features and target.
        target_column (str): The name of the target column.
        categorical_columns (list): A list of categorical column names.
        max_iter (int): Maximum number of iterations (default: 100).
        alpha (float): Significance level for feature importance (default: 0.05).

    Returns:
        pd.DataFrame: The DataFrame with the selected features (confirmed and tentative).
    """
    from boruta import BorutaPy
    from sklearn.ensemble import RandomForestRegressor
    
    # Separate features and target
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Identify numerical columns
    numerical_columns = [col for col in X.columns if col not in categorical_columns]

    # Standardize numerical columns
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X[numerical_columns]), columns=numerical_columns)

    # Initialize Boruta with RandomForest
    rf = RandomForestRegressor(n_jobs=-1, max_depth=5, random_state=42)
    boruta_selector = BorutaPy(rf, n_estimators='auto', verbose=0, random_state=42, 
                               max_iter=max_iter, alpha=alpha)

    # Fit Boruta
    boruta_selector.fit(X_scaled.values, y.values)

    # Get selected features (confirmed + tentative)
    selected_mask = boruta_selector.support_ | boruta_selector.support_weak_
    selected_features = [numerical_columns[i] for i, selected in enumerate(selected_mask) if selected]

    # Create DataFrame with selected features and target
    df_selected = pd.concat([data[selected_features], data[categorical_columns], y], axis=1)

    return df_selected

@optimized_cache
def replace_missing_with_average(data):
    """Replace missing values with the average of each column."""
    return data.fillna(data.mean())

@optimized_cache
def replace_missing_with_zero(data):
    """Replace missing values with zero."""
    return data.fillna(0)

@optimized_cache
def delete_missing_values(data):
    """Delete rows containing missing values."""
    return data.dropna()

@optimized_cache
def normalize_data_minmax(data):
    """Normalize data using Min-Max scaling."""
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(data)
    return pd.DataFrame(normalized_data, columns=data.columns)

@optimized_cache
def normalize_data_standard(data):
    """Standardize data using Z-score standardization."""
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(data)
    return pd.DataFrame(standardized_data, columns=data.columns)

@optimized_cache
def encode_categorical_onehot(data):
    """One-hot encode categorical variables."""
    categorical_columns = data.select_dtypes(include=['object']).columns
    encoder = OneHotEncoder(drop='first', sparse_output=False)
    encoded_data = pd.DataFrame(encoder.fit_transform(data[categorical_columns]))
    data = pd.concat([data, encoded_data], axis=1)
    data = data.drop(categorical_columns, axis=1)
    return data

@st.cache_data
def encode_categorical_label(data):
    """Label encode categorical variables."""
    le = LabelEncoder()
    categorical_columns = data.select_dtypes(include=['object']).columns
    for column in categorical_columns:
        data[column] = le.fit_transform(data[column])
    return data

@optimized_cache
def evaluate_regression_models(X_train, X_test, y_train, y_test):
    """
    This function evaluates various regression models using LazyRegressor.
    
    Inputs:
    - X_train: Training features (array-like, shape (n_samples, n_features))
    - X_test: Testing features (array-like, shape (n_samples, n_features))
    - y_train: Training target (array-like, shape (n_samples,))
    - y_test: Testing target (array-like, shape (n_samples,))
    
    Returns:
    - models_df: DataFrame containing information about various regression models
    - predictions_df: DataFrame containing predictions of various regression models
    """
    try:
        # Initialize LazyRegressor
        reg = LazyRegressor.LazyRegressor(verbose=1, ignore_warnings=False, custom_metric=None, predictions=True)
        
        # Fit LazyRegressor on the data
        models_df, predictions_df = reg.fit(X_train, X_test, y_train, y_test)
        
        return models_df
        
    except Exception as e:
        print(f"An error occurred: {e}")
        raise e


def tune_LassoCV_model(X_train, y_train):
    try:
        # Define the model
        lasso_cv = LassoCV()

        # Define hyperparameters to tune
        param_dist = {
            "eps": [0.001, 0.01, 0.1],
            # wide range of alpha values
            "n_alphas": [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
            "fit_intercept": [True, False],
            "precompute": ['auto', True, False],
            "max_iter": [500, 1000, 2000, 3000],
            "tol": [0.0001, 0.001, 0.01],
            "cv": [3, 5, 10],  # Cross-validation strategy
            "positive": [False, True],
            "selection": ['cyclic', 'random']
        }
        # Perform RandomizedSearchCV
        random_search = RandomizedSearchCV(lasso_cv, param_distributions=param_dist, n_iter=50, cv=5, random_state=42)
        random_search.fit(X_train, y_train)

        # Best parameters and best score
        print("Best Parameters:", random_search.best_params_)
        print("Best Score:", random_search.best_score_)

        # Return the best model
        return random_search.best_estimator_,random_search.best_score_
    
    except Exception as e:
        st.error(f"An error occurred while tuning LassoCV model: {e}")
        return None

def tune_LassoLarsCV_model(X_train, y_train):
    try:
        # Define the model
        lasso_lars_cv = LassoLarsCV()

        # Define hyperparameters to tune
        param_dist = {
            "fit_intercept": [True, False],
            "verbose": [True, False],
            "max_iter": [500, 1000, 1500],
            "precompute": [True, False, 'auto'],
            "max_n_alphas": [500, 1000, 1500],
            "eps": [1e-16, 1e-12, 1e-8],
            "copy_X": [True, False],
            "positive": [True, False]
        }

        # Perform RandomizedSearchCV
        random_search = RandomizedSearchCV(lasso_lars_cv, param_distributions=param_dist, n_iter=50, cv=5, random_state=42)
        random_search.fit(X_train, y_train)

        # Best parameters and best score
        print("Best Parameters:", random_search.best_params_)
        print("Best Score:", random_search.best_score_)

        # Return the best model
        return random_search.best_estimator_, random_search.best_score_

    except Exception as e:
        st.error(f"An error occurred while tuning LassoLarsCV model: {e}")
        return None, None

def gaussian_process_regressor_hyperparam_search(X_train, y_train):
    try:
        # Define the model
        gaussian_process = GaussianProcessRegressor()

        param_dist = {
            "kernel": [None],
            "alpha": uniform(1e-10, 1.0),
            "optimizer": ['fmin_l_bfgs_b'],
            "n_restarts_optimizer": [0],
            "normalize_y": [False],
            "copy_X_train": [True],
            "n_targets": [None],
            "random_state": [None]
        }

        # Perform RandomizedSearchCV
        random_search = RandomizedSearchCV(gaussian_process, param_distributions=param_dist, n_iter=50, cv=5, random_state=42)
        random_search.fit(X_train, y_train)

        # Best parameters and best score
        print("Best Parameters:", random_search.best_params_)
        print("Best Score:", random_search.best_score_)

        return random_search.best_estimator_, random_search.best_params_

    except Exception as e:
        print("Error occurred:", e)
        return None, None

def tune_LarsCV(X_train, y_train):
    try:
        # Define the model
        lars_cv = LarsCV()

        param_dist = {
            'fit_intercept': [True, False],
            'verbose': [False, True],
            'max_iter': randint(100, 1000),
            'precompute': ['auto', True, False],
            'cv': randint(3, 10),
            'max_n_alphas': randint(100, 2000),
            'n_jobs': [-1, None],
            'eps': uniform(1e-8, 1e-3),
            'copy_X': [True, False]
        }

        # Perform RandomizedSearchCV
        random_search = RandomizedSearchCV(lars_cv, param_distributions=param_dist, n_iter=100, cv=5, random_state=42)
        random_search.fit(X_train, y_train)

        # Best parameters and best score
        print("Best Parameters:", random_search.best_params_)
        print("Best Score:", random_search.best_score_)

        return random_search.best_estimator_, random_search.best_params_
    
    except Exception as e:
        st.error(f"An error occurred while tuning LarsCV model: {e}")
        return None, None
        
def tune_OrthogonalMatchingPursuitCV(X_train, y_train):
    try:
        # Define the model
        omp_cv = OrthogonalMatchingPursuitCV()

        param_dist = {
            "copy": [True, False],
            "fit_intercept": [True, False],
            "cv": [3, 5, 10],
        }

        # Perform RandomizedSearchCV
        random_search = RandomizedSearchCV(omp_cv, param_distributions=param_dist, n_iter=50, cv=5, random_state=42)
        random_search.fit(X_train, y_train)

        # Best parameters and best score
        print("Best Parameters:", random_search.best_params_)
        print("Best Score:", random_search.best_score_)

        return random_search.best_estimator_, random_search.best_params_

    except Exception as e:
        st.error(f"An error occurred while tuning OrthogonalMatchingPursuitCV model: {e}")
        return None, None

def tune_decision_tree_regressor(X_train, y_train):
    try:
        # Define the model
        decision_tree = DecisionTreeRegressor()

        param_dist = {
            "criterion": ["squared_error", "friedman_mse", "absolute_error", "poisson"],
            "splitter": ["best", "random"],
            "max_depth": [None] + list(range(1, 21)),
            "min_samples_split": randint(2, 11),
            "min_samples_leaf": randint(1, 11),
            "min_weight_fraction_leaf": uniform(0, 0.5),
            "max_features": [None, "sqrt", "log2"] + list(range(1, X_train.shape[1] + 1)),
            "random_state": [None],
            "max_leaf_nodes": [None] + list(range(2, 21)),
            "min_impurity_decrease": uniform(0, 0.5),
            "ccp_alpha": uniform(0, 0.5),
            "monotonic_cst": [None]
        }

        # Perform RandomizedSearchCV
        random_search = RandomizedSearchCV(decision_tree, param_distributions=param_dist, n_iter=50, cv=5, random_state=42)
        random_search.fit(X_train, y_train)

        # Best parameters and best score
        print("Best Parameters:", random_search.best_params_)
        print("Best Score:", random_search.best_score_)

        return random_search.best_estimator_, random_search.best_params_

    except Exception as e:
        print("Error occurred:", e)
        return None, None

def extra_tree_regressor_hyperparam_search(X_train, y_train):
    try:
        # Define the model
        extra_tree = ExtraTreesRegressor()

        param_dist = {
            "n_estimators": randint(10, 101),
            "criterion": ["squared_error", "absolute_error", "friedman_mse", "poisson"],
            "max_depth": [None] + list(range(1, 21)),
            "min_samples_split": randint(2, 11),
            "min_samples_leaf": randint(1, 11),
            "min_weight_fraction_leaf": uniform(0, 0.5),
            "max_features": ["sqrt", "log2", None] + list(range(1, X_train.shape[1] + 1)),
            "max_leaf_nodes": [None] + list(range(2, 21)),
            "min_impurity_decrease": uniform(0, 0.5),
            "bootstrap": [True, False],
            "oob_score": [True, False],
            "ccp_alpha": uniform(0, 0.5),
            "max_samples": [None] + list(range(1, len(X_train) + 1)),
            "random_state": [None],
            "verbose": [0],
            "n_jobs": [None],
            "warm_start": [False]
        }

        # Perform RandomizedSearchCV
        random_search = RandomizedSearchCV(extra_tree, param_distributions=param_dist, n_iter=50, cv=5, random_state=42)
        random_search.fit(X_train, y_train)

        # Best parameters and best score
        print("Best Parameters:", random_search.best_params_)
        print("Best Score:", random_search.best_score_)

        return random_search.best_estimator_, random_search.best_params_

    except Exception as e:
        print("Error occurred:", e)
        return None, None

    
def tune_NuSVR(X_train, y_train):
    try:
        # Define the model
        nusvr = NuSVR()

        param_dist = {
            "nu": uniform(0.1, 0.9),
            "C": uniform(0.1, 10),
            "kernel": ['linear', 'poly', 'rbf', 'sigmoid'],
            "degree": randint(1, 10),
            "gamma": ['scale', 'auto', uniform(0.001, 1)],
            "coef0": uniform(-1, 1),
            "shrinking": [True, False],
            "tol": uniform(1e-4, 1e-2),
            "cache_size": [100, 200, 300],
            "verbose": [True],
            "max_iter": [-1, 1000, 2000]
        }

        # Perform RandomizedSearchCV
        random_search = RandomizedSearchCV(nusvr, param_distributions=param_dist, n_iter=50, cv=5, random_state=42)
        random_search.fit(X_train, y_train)

        # Best parameters and best score
        print("Best Parameters:", random_search.best_params_)
        print("Best Score:", random_search.best_score_)

        return random_search.best_estimator_, random_search.best_params_

    except Exception as e:
        st.error(f"An error occurred while tuning NuSVR model: {e}")
        return None, None

def tune_lasso(X_train, y_train):
    try:
        # Define the model
        lasso = Lasso()

        # Define hyperparameters to tune
        param_dist = {
            "alpha": [0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
            "fit_intercept": [True, False],
            "normalize": [True, False],
            "precompute": ['auto', True, False],
            "copy_X": [True, False],
            "max_iter": [1000, 2000, 3000],
            "tol": [0.0001, 0.001, 0.01],
            "warm_start": [True, False],
            "positive": [False, True],
            "selection": ['cyclic', 'random']
        }

        # Perform RandomizedSearchCV
        random_search = RandomizedSearchCV(lasso, param_distributions=param_dist, n_iter=50, cv=5, random_state=42)
        random_search.fit(X_train, y_train)

        # Best parameters and best score
        print("Best Parameters:", random_search.best_params_)
        print("Best Score:", random_search.best_score_)

        # Return the best model
        return random_search.best_estimator_, random_search.best_params_

    except Exception as e:
        st.error(f"An error occurred while tuning Lasso model: {e}")
        return None, None
        
def tune_LassoLarsCV(X_train, y_train):
    try:
        # Define the model
        lasso_lars_cv = LassoLarsCV()

        # Define hyperparameters to tune
        param_dist = {
            "fit_intercept": [True, False],
            "verbose": [True, False],
            "normalize": [True, False],
            "precompute": [True, False, 'auto'],
            "max_iter": [500, 1000, 1500],
            "eps": [1e-16, 1e-12, 1e-8],
            "copy_X": [True, False],
            "positive": [True, False]
        }

        # Perform RandomizedSearchCV
        random_search = RandomizedSearchCV(lasso_lars_cv, param_distributions=param_dist, n_iter=50, cv=5, random_state=42)
        random_search.fit(X_train, y_train)

        # Best parameters and best score
        print("Best Parameters:", random_search.best_params_)
        print("Best Score:", random_search.best_score_)

        # Return the best model
        return random_search.best_estimator_, random_search.best_params_

    except Exception as e:
        st.error(f"An error occurred while tuning LassoLarsCV model: {e}")
        return None, None
    
def omp_hyperparam_search(X_train, y_train):
    try:
        # Define the model
        omp = OrthogonalMatchingPursuit()

        param_dist = {
                "n_nonzero_coefs": randint(1, X_train.shape[1] // 2),  # Set a reasonable range for n_nonzero_coefs
                "tol": uniform(1e-5, 1e-2),  # Set a reasonable range for tol
                "fit_intercept": [True, False],
                "precompute": ['auto', True, False],
            }

        # Perform RandomizedSearchCV
        random_search = RandomizedSearchCV(omp, param_distributions=param_dist, n_iter=50, cv=5, random_state=42)
        random_search.fit(X_train, y_train)

        # Best parameters and best score
        print("Best Parameters:", random_search.best_params_)
        print("Best Score:", random_search.best_score_)

        # Return the best model
        return random_search.best_estimator_, random_search.best_params_

    except Exception as e:
        st.error(f"An error occurred while tuning OrthogonalMatchingPursuit model: {e}")
        return None, None
        
def elastic_net_cv_hyperparam_search(X_train, y_train):
    try:
        # Define the model
        elastic_net_cv = ElasticNetCV()

        # Define hyperparameters to tune
        param_dist = {
            "l1_ratio": uniform(0, 0.99),  # Set a reasonable range for l1_ratio
            "eps": [1e-3, 1e-2, 1e-1],  # Set a reasonable range for eps
            "n_alphas": [100, 200, 300, 400, 500],  # Set a reasonable range for n_alphas
            "fit_intercept": [True, False],
            "precompute": ['auto', True, False],
            "max_iter": [500, 1000, 2000, 3000],  # Set a reasonable range for max_iter
            "tol": [1e-4, 1e-3, 1e-2],  # Set a reasonable range for tol
            "cv": [3, 5, 10],  # Set a reasonable range for cv
            "positive": [False, True],
            "selection": ['cyclic', 'random']
        }

        # Perform RandomizedSearchCV
        random_search = RandomizedSearchCV(elastic_net_cv, param_distributions=param_dist, n_iter=50, cv=5, random_state=42)
        random_search.fit(X_train, y_train)

        # Best parameters and best score
        print("Best Parameters:", random_search.best_params_)
        print("Best Score:", random_search.best_score_)

        # Return the best model
        return random_search.best_estimator_, random_search.best_params_

    except Exception as e:
        st.error(f"An error occurred while tuning ElasticNetCV model: {e}")
        return None, None

def lasso_lars_hyperparam_search(X_train, y_train):
    try:
        # Define the model
        lasso_lars = LassoLars()

        param_dist = {
            "alpha": uniform(0.1, 10.0),
            "fit_intercept": [True, False],
            "verbose": [True, False],
            "precompute": ['auto'],
            "max_iter": [500],
            "eps": [2.220446049250313e-16],
            "copy_X": [True],
            "fit_path": [True, False],
            "positive": [True, False],
            "jitter": [None],
            "random_state": [None]
        }

        # Perform RandomizedSearchCV
        random_search = RandomizedSearchCV(lasso_lars, param_distributions=param_dist, n_iter=50, cv=5, random_state=42)
        random_search.fit(X_train, y_train)

        # Best parameters and best score
        print("Best Parameters:", random_search.best_params_)
        print("Best Score:", random_search.best_score_)

        return random_search.best_estimator_, random_search.best_params_

    except Exception as e:
        print("Error occurred:", e)
        return None, None
        
def elastic_net_hyperparam_search(X_train, y_train):
    try:
        # Define the model
        elastic_net = ElasticNet()

        # Define hyperparameters to tune
        param_dist = {
            "alpha": uniform(1e-6, 1.0),  # Set a reasonable range for alpha
            "l1_ratio": uniform(0.01, 1.0),  # Set a reasonable range for l1_ratio
            "fit_intercept": [True, False],
            "precompute": ['auto', True, False],
            "max_iter": [1000, 2000, 3000],  # Set a reasonable range for max_iter
            "tol": [1e-4, 1e-3, 1e-2],  # Set a reasonable range for tol
            "warm_start": [True, False],
            "positive": [False, True],
            "selection": ['cyclic', 'random']
        }

        # Perform RandomizedSearchCV
        random_search = RandomizedSearchCV(elastic_net, param_distributions=param_dist, n_iter=50, cv=5, random_state=42)
        random_search.fit(X_train, y_train)

        # Best parameters and best score
        print("Best Parameters:", random_search.best_params_)
        print("Best Score:", random_search.best_score_)

        # Return the best model
        return random_search.best_estimator_, random_search.best_params_
    
    except Exception as e:
        st.error(f"An error occurred while tuning ElasticNet model: {e}")
        return None, None

def tweedie_regressor_hyperparam_search(X_train, y_train):
    try:
        # Define the model
        tweedie_regressor = TweedieRegressor()

        # Define hyperparameters to tune
        param_dist = {
            "power": [0, 1, 2],  # Set a reasonable range for power
            "alpha": uniform(0, 1),  # Set a reasonable range for alpha
            "fit_intercept": [True, False],
            "link": ['auto', 'identity', 'log'],
            #"solver": ["lbfgs", "newton-cholesky"],
            "warm_start": [True, False],
            "max_iter": [100, 500, 1000],  # Set a reasonable range for max_iter
            "tol": uniform(1e-5, 1e-2),  # Set a reasonable range for tol
        }

        # Perform RandomizedSearchCV
        random_search = RandomizedSearchCV(tweedie_regressor, param_distributions=param_dist, n_iter=50, cv=5, random_state=42)
        random_search.fit(X_train, y_train)

        # Best parameters and best score
        print("Best Parameters:", random_search.best_params_)
        print("Best Score:", random_search.best_score_)

        # Return the best model
        return random_search.best_estimator_, random_search.best_params_

    except Exception as e:
        st.error(f"An error occurred while tuning TweedieRegressor model: {e}")
        return None, None
        
def dummy_regressor_hyperparam_search(X_train, y_train):
    try:
        # Define the model
        dummy_regressor = DummyRegressor()

        # if y_train is numpy array
        if isinstance(y_train, np.ndarray):
            train_max, train_min, train_mean, train_median = np.max(y_train), np.min(y_train), np.mean(y_train), np.median(y_train)

        elif isinstance(y_train, pd.Series):
            train_max, train_min, train_mean, train_median = y_train.max(), y_train.min(), y_train.mean(), y_train.median()
        
        # Define hyperparameters to tune
        param_dist = {
            "strategy": ["mean", "median", "quantile", "constant"],
            "constant": [0.0, 1.0, -1.0, train_max, train_min, train_mean, train_median],
            "quantile": [0.0, 0.25, 0.5, 0.75, 1.0],
        }

        # Perform RandomizedSearchCV
        random_search = RandomizedSearchCV(dummy_regressor, param_distributions=param_dist, n_iter=50, cv=5, random_state=42)
        random_search.fit(X_train, y_train)

        # Best parameters and best score
        print("Best Parameters:", random_search.best_params_)
        print("Best Score:", random_search.best_score_)

        # Return the best model
        return random_search.best_estimator_, random_search.best_params_

    except Exception as e:
        st.error(f"An error occurred while tuning DummyRegressor model: {e}")
        return None, None
        
def huber_regressor_hyperparam_search(X_train, y_train):
    try:
        # Define the model
        huber_regressor = HuberRegressor()

        # Define hyperparameters to tune
        param_dist = {
            "epsilon": uniform(1.0, 2.0),  # Epsilon controls the number of outliers
            "alpha": uniform(1e-6, 1.0),  # Strength of L2 regularization
            "max_iter": [100, 500, 1000],  # Maximum number of iterations
            "fit_intercept": [True, False],
            "tol": uniform(1e-5, 1e-2),  # Tolerance for stopping criterion
        }

        # Perform RandomizedSearchCV
        random_search = RandomizedSearchCV(huber_regressor, param_distributions=param_dist, n_iter=50, cv=5, random_state=42)
        random_search.fit(X_train, y_train)

        # Best parameters and best score
        print("Best Parameters:", random_search.best_params_)
        print("Best Score:", random_search.best_score_)

        # Return the best model
        return random_search.best_estimator_, random_search.best_params_

    except Exception as e:
        st.error(f"An error occurred while tuning HuberRegressor model: {e}")
        return None, None
        

def svr_hyperparam_search(X_train, y_train, scoring='r2'):
    try:
        # Define the model
        svr = NuSVR()

        # Define hyperparameters to tune
        param_dist = {
            "nu": uniform(0.1, 0.9),  # An upper bound on the fraction of margin errors and a lower bound of support vectors
            "C": uniform(0.1, 100),  # Regularization parameter
            "kernel": ['linear', 'poly', 'rbf', 'sigmoid'],  # Type of kernel
            "degree": randint(1, 10),  # Degree of the polynomial kernel
            "gamma": ['scale', 'auto', uniform(0.001, 1)],  # Kernel coefficient for 'rbf', 'poly' and 'sigmoid'
            "coef0": uniform(-1, 1),  # Independent term in kernel function
            "shrinking": [True, False],  # Whether to use the shrinking heuristic
            "tol": uniform(1e-4, 1e-2),  # Tolerance for stopping criterion
            "cache_size": [100, 200, 300],  # Size of the kernel cache
        }

        # Perform RandomizedSearchCV with custom scoring metric
        random_search = RandomizedSearchCV(svr, param_distributions=param_dist, n_iter=50, cv=5, scoring=scoring, random_state=42)
        random_search.fit(X_train, y_train)

        # Best parameters and best score
        print("Best Parameters:", random_search.best_params_)
        print("Best Score:", random_search.best_score_)

        # Return the best model
        return random_search.best_estimator_, random_search.best_params_

    except Exception as e:
        st.error(f"An error occurred while tuning SVR model: {e}")
        return None, None
    

def ransac_regressor_hyperparam_search(X_train, y_train):
    try:
        # Define the model
        ransac_regressor = RANSACRegressor()

        # Define hyperparameters to tune
        param_dist = {
            "min_samples": uniform(0.1, 0.5),  # Minimum number of samples for consensus set
            "residual_threshold": uniform(1.0, 5.0),  # Maximum residual for inliers
            "max_trials": [100, 200, 300],  # Maximum number of iterations for random sample selection
            "max_skips": [10, 20, 30],  # Maximum number of iterations that can be skipped
            "stop_n_inliers": [int(0.8 * X_train.shape[0]), int(0.9 * X_train.shape[0])],  # Stop iteration if at least this number of inliers are found
            "stop_score": [0.95, 0.96, 0.97],  # Stop iteration if score is greater equal than this threshold
            "stop_probability": [0.98, 0.99],  # Probability for stopping iteration
            "loss": ["absolute_loss", "squared_loss"],  # Loss function for determining inliers/outliers
        }

        # Perform RandomizedSearchCV
        random_search = RandomizedSearchCV(ransac_regressor, param_distributions=param_dist, n_iter=50, cv=5, random_state=42)
        random_search.fit(X_train, y_train)

        # Best parameters and best score
        print("Best Parameters:", random_search.best_params_)
        print("Best Score:", random_search.best_score_)

        # Return the best model
        return random_search.best_estimator_, random_search.best_params_
    
    except Exception as e:
        st.error(f"An error occurred while tuning RANSACRegressor model: {e}")
        return None, None

def ridge_cv_hyperparam_search(X_train, y_train):
    try:
        # Define the model
        ridge_cv = RidgeCV()

        # Define parameter distributions
        param_dist = {
            "alphas": [(0.1, 1.0, 10.0), uniform(0.1, 10.0)]  # Example distribution for alphas
            # Add more parameters and distributions as needed
        }

        # Perform RandomizedSearchCV
        random_search = RandomizedSearchCV(ridge_cv, param_distributions=param_dist, n_iter=50, cv=5, random_state=42)
        random_search.fit(X_train, y_train)

        # Return best estimator and best parameters
        return random_search.best_estimator_, random_search.best_params_

    except Exception as e:
        print("Error occurred:", e)
        return None, None

def bayesian_ridge_hyperparam_search(X_train, y_train):
    try:

        # Define the model
        bayesian_ridge = BayesianRidge()

        # Define hyperparameters to tune
        param_dist = {
            "tol": uniform(1e-5, 1e-2),  # Tolerance for stopping criterion
            "alpha_1": uniform(1e-8, 1e-4),  # Shape parameter for Gamma prior over alpha
            "alpha_2": uniform(1e-8, 1e-4),  # Inverse scale parameter for Gamma prior over alpha
            "lambda_1": uniform(1e-8, 1e-4),  # Shape parameter for Gamma prior over lambda
            "lambda_2": uniform(1e-8, 1e-4),  # Inverse scale parameter for Gamma prior over lambda
            "alpha_init": [None] + list(uniform(0.1, 1.0).rvs(10)),  # Initial value for alpha
            "lambda_init": [None] + list(uniform(0.1, 1.0).rvs(10)),  # Initial value for lambda
            "compute_score": [True, False],
            "fit_intercept": [True, False],
            "copy_X": [True, False],
            "verbose": [False],
        }

        # Perform RandomizedSearchCV
        random_search = RandomizedSearchCV(bayesian_ridge, param_distributions=param_dist, n_iter=50, cv=5, random_state=42)
        random_search.fit(X_train, y_train)

        # Best parameters and best score
        print("Best Parameters:", random_search.best_params_)
        print("Best Score:", random_search.best_score_)

        # Return the best model
        return random_search.best_estimator_, random_search.best_params_
    
    except Exception as e:
        st.error(f"An error occurred while tuning BayesianRidge model: {e}")
        return None, None

def ridge_hyperparam_search(X_train, y_train):
    try:
        # Define the model
        ridge = Ridge()

        # Define hyperparameters to tune
        param_dist = {
            "alpha": uniform(0.1, 10),  # Regularization strength
            "fit_intercept": [True, False],
            "copy_X": [True, False],
            "max_iter": [100, 500, 1000],  # Maximum number of iterations
            "tol": uniform(1e-5, 1e-2),  # Tolerance for stopping criterion
            "solver": ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"],
        }

        # Perform RandomizedSearchCV
        random_search = RandomizedSearchCV(ridge, param_distributions=param_dist, n_iter=50, cv=5, random_state=42)
        random_search.fit(X_train, y_train)

        # Best parameters and best score
        print("Best Parameters:", random_search.best_params_)
        print("Best Score:", random_search.best_score_)

        # Return the best model
        return random_search.best_estimator_, random_search.best_params_
    
    except Exception as e:
        st.error(f"An error occurred while tuning Ridge model: {e}")
        return None, None
        
def linear_regression_hyperparam_search(X_train, y_train):
    try:
        # Define the model
        linear_regression = LinearRegression()

        # Define hyperparameters to tune
        param_dist = {
            "fit_intercept": [True, False],
            "copy_X": [True, False],
        }

        # Perform RandomizedSearchCV
        random_search = RandomizedSearchCV(linear_regression, param_distributions=param_dist, n_iter=50, cv=5, random_state=42)
        random_search.fit(X_train, y_train)

        # Best parameters and best score
        print("Best Parameters:", random_search.best_params_)
        print("Best Score:", random_search.best_score_)

        # Return the best model
        return random_search.best_estimator_, random_search.best_params_

    except Exception as e:
        st.error(f"An error occurred while tuning LinearRegression model: {e}")
        return None, None
        
def transformed_target_regressor_hyperparam_search(X_train, y_train):
    try:
        # Define the model
        transformed_target_regressor = TransformedTargetRegressor()

        # Define hyperparameters to tune
        param_dist = {
            "regressor": [LinearRegression(), Ridge(), BayesianRidge()],
            "transformer": [None, "quantile", "yeo-johnson", "box-cox"],
        }

        # Perform RandomizedSearchCV
        random_search = RandomizedSearchCV(transformed_target_regressor, param_distributions=param_dist, n_iter=50, cv=5, random_state=42)
        random_search.fit(X_train, y_train)

        # Best parameters and best score
        print("Best Parameters:", random_search.best_params_)
        print("Best Score:", random_search.best_score_)

        # Return the best model
        return random_search.best_estimator_, random_search.best_params_

    except Exception as e:
        st.error(f"An error occurred while tuning TransformedTargetRegressor model: {e}")
        return None, None
        
def lasso_lars_ic_hyperparam_search(X_train, y_train):
    try:
        # Define the model
        lasso_lars_ic = LassoLarsIC()

        # Define hyperparameters to tune
        param_dist = {
            "criterion": ["aic", "bic"],
            "fit_intercept": [True, False],
            "max_iter": [500, 1000, 1500],
            "eps": [1e-16, 1e-12, 1e-8],
        }

        # Perform RandomizedSearchCV
        random_search = RandomizedSearchCV(lasso_lars_ic, param_distributions=param_dist, n_iter=50, cv=5, random_state=42)
        random_search.fit(X_train, y_train)

        # Best parameters and best score
        print("Best Parameters:", random_search.best_params_)
        print("Best Score:", random_search.best_score_)

        # Return the best model
        return random_search.best_estimator_, random_search.best_params_

    except Exception as e:
        st.error(f"An error occurred while tuning LassoLarsIC model: {e}")
        return None, None
        
def lars_hyperparam_search(X_train, y_train):
    try:
        # Define the model
        lars = Lars()

        # Define hyperparameters to tune
        param_dist = {
            "fit_intercept": [True, False],
            "precompute": [True, False, 'auto'],
            "n_nonzero_coefs": [100, 200, 300, 400, 500],
            "eps": uniform(1e-16, 1e-8),
            "copy_X": [True, False],
        }

        # Perform RandomizedSearchCV
        random_search = RandomizedSearchCV(lars, param_distributions=param_dist, n_iter=50, cv=5, random_state=42)
        random_search.fit(X_train, y_train)

        # Best parameters and best score
        print("Best Parameters:", random_search.best_params_)
        print("Best Score:", random_search.best_score_)

        return random_search.best_estimator_, random_search.best_params_

    except Exception as e:
        st.error(f"An error occurred while tuning Lars model: {e}")
        return None, None


def mlp_regressor_hyperparam_search(X_train, y_train):
    try:
        # Define the model
        mlp_regressor = MLPRegressor()

        # Define hyperparameters to tune
        param_dist = {
            "hidden_layer_sizes": [(100,), (200,), (300,), (400,), (500,)],
            "activation": ["identity", "logistic", "tanh", "relu"],
            "solver": ["lbfgs", "sgd", "adam"],
            "alpha": loguniform(1e-6, 1.0),
            "batch_size": [16, 32, 64, 128],
            "learning_rate": ["constant", "invscaling", "adaptive"],
            "learning_rate_init": loguniform(1e-4, 1e-2),
            "power_t": loguniform(0.1, 1.0),
            "max_iter": [200, 400, 600, 800, 1000],
            "shuffle": [True, False],
            "tol": loguniform(1e-5, 1e-2),
            "warm_start": [True, False],
            "momentum": np.linspace(0.1, 0.9, 20),
            "nesterovs_momentum": [True, False],
            "early_stopping": [True, False],
            "validation_fraction": loguniform(0.1, 0.3),
            "beta_1": np.linspace(0.1, 0.9, 20),
            "beta_2": np.linspace(0.1, 0.9, 20),
            "epsilon": loguniform(1e-8, 1e-4),
            "n_iter_no_change": [10, 20, 30],
            "max_fun": [15000],
        }

        # Perform RandomizedSearchCV
        random_search = RandomizedSearchCV(mlp_regressor, param_distributions=param_dist, n_iter=50, cv=5, random_state=42)
        random_search.fit(X_train, y_train)

        # Best parameters and best score
        print("Best Parameters:", random_search.best_params_)
        print("Best Score:", random_search.best_score_)

        return random_search.best_estimator_, random_search.best_params_

    except Exception as e:
        st.error(f"An error occurred while tuning MLPRegressor model: {e}")
        return None, None
        

def knn_regressor_hyperparam_search(X_train, y_train):
    try:
        # Define the model
        knn_regressor = KNeighborsRegressor()

        # Define hyperparameters to tune
        param_dist = {
            "n_neighbors": [3, 5, 7, 9, 11],
            "weights": ["uniform", "distance"],
            "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
            "leaf_size": [10, 20, 30, 40, 50],
            "p": [1, 2],
        }

        # Perform RandomizedSearchCV
        random_search = RandomizedSearchCV(knn_regressor, param_distributions=param_dist, n_iter=50, cv=5, random_state=42)
        random_search.fit(X_train, y_train)

        # Best parameters and best score
        print("Best Parameters:", random_search.best_params_)
        print("Best Score:", random_search.best_score_)

        return random_search.best_estimator_, random_search.best_params_

    except Exception as e:
        st.error(f"An error occurred while tuning KNeighborsRegressor model: {e}")
        return None, None
    
def extra_trees_regressor_hyperparam_search(X_train, y_train):
    try:
        # Define the model
        extra_trees_regressor = ExtraTreesRegressor()

        # Define hyperparameters to tune
        param_dist = {
            "n_estimators": [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
            "criterion": ["squared_error", "absolute_error", "friedman_mse", "poisson"],
            "max_depth": [10, 20, 30, 40, 50],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "min_weight_fraction_leaf": uniform(0.0, 0.5),
            "max_features": ["auto", "sqrt", "log2"],
            "max_leaf_nodes": [None, 10, 20, 30],
            "min_impurity_decrease": uniform(0.0, 0.5),
            "bootstrap": [True, False],
            "oob_score": [True, False],
            "random_state": [None, 42],
            "warm_start": [True, False],
        }

        # Perform RandomizedSearchCV
        random_search = RandomizedSearchCV(extra_trees_regressor, param_distributions=param_dist, n_iter=50, cv=5, random_state=42)
        random_search.fit(X_train, y_train)

        # Best parameters and best score
        print("Best Parameters:", random_search.best_params_)
        print("Best Score:", random_search.best_score_)

        return random_search.best_estimator_, random_search.best_params_

    except Exception as e:
        st.error(f"An error occurred while tuning ExtraTreesRegressor model: {e}")
        return None, None
        
def kernel_ridge_hyperparam_search(X_train, y_train):
    try:
        # Define the model
        kernel_ridge = KernelRidge()

        # Define hyperparameters to tune
        param_dist = {
            "alpha": uniform(1e-6, 1.0),
            "kernel": ["linear", "poly", "rbf", "sigmoid"],
            "degree": randint(1, 10),
            "gamma": ["scale", "auto", uniform(0.001, 1)],
            "coef0": uniform(-1, 1),
        }

        # Perform RandomizedSearchCV
        random_search = RandomizedSearchCV(kernel_ridge, param_distributions=param_dist, n_iter=50, cv=5, random_state=42)
        random_search.fit(X_train, y_train)

        # Best parameters and best score
        print("Best Parameters:", random_search.best_params_)
        print("Best Score:", random_search.best_score_)

        return random_search.best_estimator_, random_search.best_params_

    except Exception as e:
        st.error(f"An error occurred while tuning KernelRidge model: {e}")
        return None, None
    
def ada_boost_regressor_hyperparam_search(X_train, y_train):
    try:
        # Define the model
        ada_boost_regressor = AdaBoostRegressor()

        # Define hyperparameters to tune
        param_dist = {
            "base_estimator": [None, LinearRegression(), Ridge(), BayesianRidge()],
            "n_estimators": [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
            "learning_rate": uniform(0.1, 1.0),
            "loss": ["linear", "square", "exponential"],
            "random_state": [42],
        }

        # Perform RandomizedSearchCV
        random_search = RandomizedSearchCV(ada_boost_regressor, param_distributions=param_dist, n_iter=50, cv=5, random_state=42)
        random_search.fit(X_train, y_train)

        # Best parameters and best score
        print("Best Parameters:", random_search.best_params_)
        print("Best Score:", random_search.best_score_)

        return random_search.best_estimator_, random_search.best_params_

    except Exception as e:
        st.error(f"An error occurred while tuning AdaBoostRegressor model: {e}")
        return None, None
        

def passive_aggressive_regressor_hyperparam_search(X_train, y_train):
    try:
        # Define the model
        passive_aggressive_regressor = PassiveAggressiveRegressor()

        # Define hyperparameters to tune
        param_dist = {
            "C": uniform(0.1, 10),
            "fit_intercept": [True, False],
            "max_iter": [1000, 2000, 3000],
            "tol": uniform(1e-5, 1e-2),
            "early_stopping": [True, False],
            "validation_fraction": uniform(0.1, 0.3),
            "n_iter_no_change": [5, 10, 15],
            "shuffle": [True, False],
            "loss": ["epsilon_insensitive", "squared_epsilon_insensitive"],
            "random_state": [42],
            "warm_start": [True, False],
        }

        # Perform RandomizedSearchCV
        random_search = RandomizedSearchCV(passive_aggressive_regressor, param_distributions=param_dist, n_iter=50, cv=5, random_state=42)
        random_search.fit(X_train, y_train)

        # Best parameters and best score
        print("Best Parameters:", random_search.best_params_)
        print("Best Score:", random_search.best_score_)

        return random_search.best_estimator_, random_search.best_params_

    except Exception as e:
        st.error(f"An error occurred while tuning PassiveAggressiveRegressor model: {e}")
        return None, None
        
def gradient_boosting_regressor_hyperparam_search(X_train, y_train, scoring='r2'):
    try:
        # Define the model
        gradient_boosting_regressor = GradientBoostingRegressor()

        # Define hyperparameters to tune
        param_dist = {
            "loss": ["ls", "lad", "huber", "quantile"],
            "learning_rate": [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "n_estimators": [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
            "subsample": uniform(0.1, 1.0),
            "criterion": ['squared_error', 'poisson', 'absolute_error', 'friedman_mse'],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "min_weight_fraction_leaf": uniform(0.0, 0.5),
            "max_depth": [3, 4, 5, 6, 7],
            "min_impurity_decrease": uniform(0.0, 0.5),
            "max_features": ["auto", "sqrt", "log2"],
            "max_leaf_nodes": [None, 10, 20, 30],
            "warm_start": [True, False],
            "validation_fraction": uniform(0.1, 0.3),
            "n_iter_no_change": [5, 10, 15],
            "tol": uniform(1e-5, 1e-2),
            "random_state": [42],
        }

        # Perform RandomizedSearchCV with custom scoring metric
        random_search = RandomizedSearchCV(gradient_boosting_regressor, param_distributions=param_dist, n_iter=50, cv=5, scoring=scoring, random_state=42)
        random_search.fit(X_train, y_train)

        # Best parameters and best score
        print("Best Parameters:", random_search.best_params_)
        print("Best Score:", random_search.best_score_)

        return random_search.best_estimator_, random_search.best_params_

    except Exception as e:
        st.error(f"An error occurred while tuning GradientBoostingRegressor model: {e}")
        return None, None

def tune_sgd_regressor(X_train, y_train):
    try:
        # Define the model
        sgd_regressor = SGDRegressor()

        # Define hyperparameters to tune
        param_dist = {
            "loss": ["squared_loss", "huber", "epsilon_insensitive", "squared_epsilon_insensitive"],
            "penalty": ["l2", "l1", "elasticnet"],
            "alpha": uniform(1e-6, 1.0),
            "l1_ratio": uniform(0.01, 0.99),
            "fit_intercept": [True, False],
            "max_iter": [1000, 2000, 3000],
            "tol": uniform(1e-5, 1e-2),
            "shuffle": [True, False],
            "epsilon": uniform(0.1, 1.0),
            "learning_rate": ["constant", "optimal", "invscaling", "adaptive"],
            "eta0": uniform(0.01, 1.0),
            "power_t": uniform(0.1, 1.0),
            "early_stopping": [True, False],
            "validation_fraction": uniform(0.1, 0.3),
            "n_iter_no_change": [5, 10, 15],
            "warm_start": [True, False],
            "average": [True, False],
        }

        # Perform RandomizedSearchCV
        random_search = RandomizedSearchCV(sgd_regressor, param_distributions=param_dist, n_iter=50, cv=5, random_state=42)
        random_search.fit(X_train, y_train)

        # Best parameters and best score
        print("Best Parameters:", random_search.best_params_)
        print("Best Score:", random_search.best_score_)

        return random_search.best_estimator_, random_search.best_params_

    except Exception as e:
        st.error(f"An error occurred while tuning SGDRegressor model: {e}")
        return None, None
    

def tune_rf_regressor(X_train, y_train, scoring='r2'):
    try:
        # Define the model
        rf_regressor = RandomForestRegressor()

        # Define hyperparameters to tune
        param_dist = {
            "n_estimators": [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)],
            "criterion": ['squared_error', 'poisson', 'absolute_error', 'friedman_mse'],
            "max_depth": [10, 20, 30, 40, 50],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "min_weight_fraction_leaf": uniform(0.0, 0.5),
            "max_features": ["auto", "sqrt", "log2"],
            "max_leaf_nodes": [None, 10, 20, 30],
            "min_impurity_decrease": uniform(0.0, 0.5),
            "bootstrap": [True, False],
            "oob_score": [True, False],
            "warm_start": [True],
        }

        # Perform RandomizedSearchCV with custom scoring metric
        random_search = RandomizedSearchCV(rf_regressor, param_distributions=param_dist, n_iter=50, cv=5, scoring=scoring, random_state=42)
        random_search.fit(X_train, y_train)

        # Best parameters and best score
        print("Best Parameters:", random_search.best_params_)
        print("Best Score:", random_search.best_score_)

        return random_search.best_estimator_, random_search.best_params_

    except Exception as e:
        st.error(f"An error occurred while tuning RandomForestRegressor model: {e}")
        return None, None
        
def tune_hist_gradient_boosting_regressor(X_train, y_train, scoring='r2'):
    try:
        # Define the model
        hist_gbr = HistGradientBoostingRegressor()

        # Define hyperparameters to tune
        param_dist = {
            'loss': ['squared_error', 'absolute_error', 'gamma', 'poisson'],
            'learning_rate': uniform(0.01, 0.5),
            'max_iter': randint(50, 500),
            'max_leaf_nodes': randint(10, 100),
            'max_depth': [None] + list(randint(2, 20)),
            'min_samples_leaf': randint(5, 50),
            'l2_regularization': uniform(0, 1),
            'max_features': uniform(0.1, 1.0),
            'max_bins': randint(32, 256),
            'categorical_features': ['auto', 'from_dtype', None],
            'early_stopping': ['auto', True, False],
            'n_iter_no_change': randint(5, 20),
            'validation_fraction': uniform(0.1, 0.3),
            'tol': [1e-8, 1e-7, 1e-6, 1e-5],
        }

        # Perform RandomizedSearchCV with custom scoring metric
        random_search = RandomizedSearchCV(hist_gbr, param_distributions=param_dist, n_iter=50, cv=5, scoring=scoring, random_state=42)
        random_search.fit(X_train, y_train)

        # Best parameters and best score
        print("Best Parameters:", random_search.best_params_)
        print("Best Score:", random_search.best_score_)

        return random_search.best_estimator_, random_search.best_params_

    except Exception as e:
        st.error(f"An error occurred while tuning HistGradientBoostingRegressor model: {e}")
        return None, None
        

def tune_bagging_regressor(X_train, y_train):
    try:
        # Define the model
        bagging_regressor = BaggingRegressor()

        # Define hyperparameters to tune
        param_dist = {
            'n_estimators': randint(10, 200),
            'max_samples': uniform(0.1, 1.0),
            'max_features': uniform(0.1, 1.0),
            'bootstrap': [True, False],
            'bootstrap_features': [True, False],
            'oob_score': [True, False],
            'warm_start': [True, False],
        }

        # Perform RandomizedSearchCV
        random_search = RandomizedSearchCV(bagging_regressor, param_distributions=param_dist, n_iter=50, cv=5, random_state=42)
        random_search.fit(X_train, y_train)

        # Best parameters and best score
        print("Best Parameters:", random_search.best_params_)
        print("Best Score:", random_search.best_score_)

        return random_search.best_estimator_, random_search.best_params_
    
    except Exception as e:
        st.error(f"An error occurred while tuning BaggingRegressor model: {e}")
        return None, None
        
def tune_lgbm_regressor(X_train, y_train, scoring='r2'):
    try:
        
        # Define the model
        lgbm_regressor = LGBMRegressor()
        
        # Define hyperparameters to tune
        param_dist = {
            'boosting_type': ['gbdt', 'dart', 'rf'],
            'num_leaves': randint(10, 200),
            'max_depth': randint(-1, 20),
            'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
            'n_estimators': randint(100, 1000),
            'subsample_for_bin': randint(20000, 300000),
            'min_split_gain': uniform(0, 1),
            'min_child_weight': uniform(1e-5, 1e-1),
            'min_child_samples': randint(5, 100),
            'subsample': uniform(0.5, 1),
            'subsample_freq': randint(0, 10),
            'colsample_bytree': uniform(0.5, 1),
            'reg_alpha': uniform(0, 1),
            'reg_lambda': uniform(0, 1),
            'random_state': randint(1, 1000)
        }
        
        # Perform RandomizedSearchCV with custom scoring metric
        random_search = RandomizedSearchCV(lgbm_regressor, param_distributions=param_dist,
                                        n_iter=100, cv=5, scoring=scoring, random_state=42, n_jobs=-1)
        random_search.fit(X_train, y_train)
        
        # Best parameters and best score
        best_model = random_search.best_estimator_
        best_params = random_search.best_params_

        return best_model, best_params
    
    except Exception as e:
        st.error(f"An error occurred while tuning LGBMRegressor model: {e}")
        return None, None
        

def tune_xgb_regressor(X_train, y_train, scoring='r2'):
    try:

        # Define the parameter grid for the search
        param_grid = {
            'max_depth': [4, 5, 6, 7, 8],
            'learning_rate': [0.1, 0.01, 0.001],
            'n_estimators': [i for i in range(100, 1400, 150)],
            'subsample': [i/10 for i in range(1, 10)],
            'colsample_bytree': [i/10 for i in range(1, 10)],
            'reg_alpha': [i/10 for i in range(1, 10)],
            'reg_lambda': [i/10 for i in range(1, 10)],
            'min_child_weight': [i for i in range(2, 8)]
        }
        
        # Initialize the XGBoost model
        xgb_model = xgb.XGBRegressor()
        
        # Perform the randomized search with cross-validation using custom scoring metric
        search = RandomizedSearchCV(xgb_model, param_grid, cv=3, n_iter=10, scoring=scoring, random_state=42)
        search.fit(X_train, y_train)
        
        # Get the best model, parameters, and score
        best_model = search.best_estimator_
        best_params = search.best_params_
        best_score = search.best_score_
        
        return best_model, best_params
    
    except Exception as e:
        st.error(f"An error occurred while tuning XGBRegressor model: {e}")
        return None, None

@optimized_cache
def evaluate_model(_best_model, X_train, y_train, X_test, y_test):
    try:
        model_name = str(_best_model).split('(')[0]
        y_test_pred = _best_model.predict(X_test)
        
        metrics = {
            "MAE": mean_absolute_error,
            "MSE": mean_squared_error,
            "R2": r2_score,
        }
        
        results = {name: metric(y_test, y_test_pred) for name, metric in metrics.items()}
        results["RMSE"] = np.sqrt(results["MSE"])
        results["RPD"] = y_test.std() / results["RMSE"]
        
        results_df = pd.DataFrame(results, index=[model_name])
        
        model_evaluation = {
            "y_test": y_test,
            "y_test_pred": y_test_pred
        }
        
        return results_df, model_evaluation
    except Exception as e:
        st.error(f"An error occurred while evaluating the model: {str(e)}")
        return pd.DataFrame(), {}
        
@st.cache_data
def plot_scatter_subplot(model_evaluation):
    try:
        y_test = model_evaluation["y_test"]
        y_test_pred = model_evaluation["y_test_pred"]
        
        fig = go.Figure()

        scatter_trace = go.Scatter(
            x=y_test_pred,
            y=y_test,
            mode='markers',
            marker=dict(color='blue', line=dict(color='black', width=1)),
            name="Predictions vs True Values"
        )

        min_val = min(np.min(y_test), np.min(y_test_pred))
        max_val = max(np.max(y_test), np.max(y_test_pred))
        
        reference_line = go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            line=dict(color='red', dash='dash'),
            name="Perfect Prediction"
        )

        fig.add_trace(scatter_trace)
        fig.add_trace(reference_line)

        fig.update_layout(
            title="Model Predictions vs True Values",
            xaxis_title="Predicted Values",
            yaxis_title="True Values",
            legend_title="Legend",
            font=dict(size=12)
        )

        st.plotly_chart(fig)

    except Exception as e:
        st.error(f"An error occurred while plotting scatter subplot: {str(e)}")


@optimized_cache
def plot_feature_importance(_best_model, X_train, y_train):
    try:
        if hasattr(best_model, 'feature_importances_'):  # For models with feature_importances_
            importances = best_model.feature_importances_
        else:  # For models without feature_importances_
            result = permutation_importance(best_model, X_train, y_train, n_repeats=10, random_state=42)
            importances = result.importances_mean
    
        indices = np.argsort(importances)[::-1]
        names = [X_train.columns[i] for i in indices]
        importance_values = [importances[i] for i in indices]
    
        fig = go.Figure(go.Pie(labels=names, values=importance_values,
                                textinfo='label+percent', hole=0.3,
                                marker=dict(colors=plt.cm.tab20c.colors, line=dict(color='white', width=2))))
    
        fig.update_layout(title_text="Feature Importance", margin=dict(l=0, r=0, t=60, b=0))
        st.plotly_chart(fig)
    except Exception as e:
        st.error(f"An error occurred while plotting feature importance: {e}")

@optimized_cache
def plot_pdp(_best_model, X_train, features, target_column):
    try:
        colors = ['#2a9d8f', '#e76f51', '#f4a261', '#738bd7', '#d35400', '#a6c7d8']
        num_features = len(features)
        max_plots_per_row = 5
        num_rows = (num_features + max_plots_per_row - 1) // max_plots_per_row
        num_cols = min(num_features, max_plots_per_row)

        fig, axs = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 5 * num_rows), constrained_layout=True)

        for j, selected_feature in enumerate(features):
            row_idx = j // max_plots_per_row
            col_idx = j % max_plots_per_row

            features_info = {
                "features": [selected_feature],
                "kind": "average",
            }

            if num_rows == 1:  # If only one row, axs is 1D
                display = PartialDependenceDisplay.from_estimator(
                    best_model,
                    X_train,
                    **features_info,
                    ax=axs[col_idx],
                )
            else:
                display = PartialDependenceDisplay.from_estimator(
                    best_model,
                    X_train,
                    **features_info,
                    ax=axs[row_idx, col_idx],
                )

            color_idx = j % len(colors)
            axs[row_idx, col_idx].set_facecolor(colors[color_idx])
            axs[row_idx, col_idx].set_title(f"PDP for {selected_feature}")
            axs[row_idx, col_idx].set_xlabel(selected_feature)
            axs[row_idx, col_idx].set_ylabel(f"Partial Dependence for {target_column}")

        fig.suptitle(f"Partial Dependence of {target_column} on Selected Features", y=1.02)
        plt.tight_layout()
        st.pyplot(fig)

    except Exception as e:
        st.error(f"An error occurred while plotting partial dependence: {e}")



# Create dict for model names and functions
model_functions_dict = {
    'ExtraTreesRegressor': extra_trees_regressor_hyperparam_search,
    'RandomForestRegressor': tune_rf_regressor,
    'BaggingRegressor': tune_bagging_regressor,
    'HistGradientBoostingRegressor': tune_hist_gradient_boosting_regressor,
    'XGBRegressor': tune_xgb_regressor,
    'GradientBoostingRegressor': gradient_boosting_regressor_hyperparam_search,
    'LGBMRegressor': tune_lgbm_regressor,
    'DecisionTreeRegressor': tune_decision_tree_regressor,
    'KNeighborsRegressor': knn_regressor_hyperparam_search,
    'ExtraTreeRegressor': extra_tree_regressor_hyperparam_search,
    'AdaBoostRegressor': ada_boost_regressor_hyperparam_search,
    'SVR': svr_hyperparam_search,
    'MLPRegressor': mlp_regressor_hyperparam_search,
    'NuSVR': tune_NuSVR,
    'BayesianRidge': bayesian_ridge_hyperparam_search,
    'ElasticNetCV': elastic_net_cv_hyperparam_search,
    'RidgeCV': ridge_cv_hyperparam_search,
    'LassoCV': tune_LassoCV_model,
    'Ridge': ridge_hyperparam_search,
    'LassoLarsCV': tune_LassoLarsCV_model,
    'LassoLarsIC': lasso_lars_ic_hyperparam_search,
    'LarsCV': tune_LarsCV,
    'OrthogonalMatchingPursuitCV': tune_OrthogonalMatchingPursuitCV,
    'TransformedTargetRegressor': transformed_target_regressor_hyperparam_search,
    'Lars': lars_hyperparam_search,
    'LinearRegression': linear_regression_hyperparam_search,
    'SGDRegressor': tune_sgd_regressor,
    'HuberRegressor': huber_regressor_hyperparam_search,
    'LinearSVR': svr_hyperparam_search,
    'TweedieRegressor': tweedie_regressor_hyperparam_search,
    'OrthogonalMatchingPursuit': omp_hyperparam_search,
    'ElasticNet': elastic_net_hyperparam_search,
    'RANSACRegressor': ransac_regressor_hyperparam_search,
    'LassoLars': lasso_lars_hyperparam_search,
    'DummyRegressor': dummy_regressor_hyperparam_search,
    'Lasso': tune_lasso,
    'PassiveAggressiveRegressor': passive_aggressive_regressor_hyperparam_search,
    'KernelRidge': kernel_ridge_hyperparam_search,
    'GaussianProcessRegressor': gaussian_process_regressor_hyperparam_search
}

# Missing Data Handling Functions
def impute_missing_values(data, method='mean', k=5):
    """
    Impute missing values in a DataFrame using various methods.
    
    Args:
        data (pd.DataFrame): Input DataFrame with missing values
        method (str): Imputation method ('mean', 'median', 'mode', 'knn', 'drop_rows', 'drop_cols')
        k (int): Number of neighbors for KNN imputation
        
    Returns:
        pd.DataFrame: DataFrame with imputed values
    """
    if method == 'mean':
        return data.fillna(data.mean())
    elif method == 'median':
        return data.fillna(data.median())
    elif method == 'mode':
        return data.fillna(data.mode().iloc[0])
    elif method == 'knn':
        from sklearn.impute import KNNImputer
        imputer = KNNImputer(n_neighbors=k)
        return pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
    elif method == 'drop_rows':
        return data.dropna()
    elif method == 'drop_cols':
        return data.dropna(axis=1)
    else:
        raise ValueError(f"Unknown imputation method: {method}")

# Data Smoothing Functions
def smooth_data(data, method='moving_average', window_size=3, alpha=0.3):
    """
    Apply smoothing to numerical data.
    
    Args:
        data (pd.DataFrame): Input DataFrame
        method (str): Smoothing method ('moving_average', 'exponential', 'savgol', 'lowess')
        window_size (int): Window size for moving average and Savitzky-Golay
        alpha (float): Smoothing factor for exponential smoothing
        
    Returns:
        pd.DataFrame: Smoothed DataFrame
    """
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    result = data.copy()
    
    if method == 'moving_average':
        result[numeric_cols] = result[numeric_cols].rolling(window=window_size, center=True).mean()
    elif method == 'exponential':
        result[numeric_cols] = result[numeric_cols].ewm(alpha=alpha).mean()
    elif method == 'savgol':
        from scipy.signal import savgol_filter
        for col in numeric_cols:
            result[col] = savgol_filter(data[col], window_size, 3)
    elif method == 'lowess':
        from statsmodels.nonparametric.smoothers_lowess import lowess
        for col in numeric_cols:
            result[col] = lowess(data[col], np.arange(len(data[col])), frac=window_size/len(data))[:, 1]
    else:
        raise ValueError(f"Unknown smoothing method: {method}")
    
    return result

# Outlier Removal Functions
def remove_outliers_iqr(df, columns, multiplier=1.5):
    """
    Remove outliers using the Interquartile Range (IQR) method on specified columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list): List of column names to check for outliers
        multiplier (float): IQR multiplier for outlier threshold (default 1.5)
        
    Returns:
        pd.DataFrame: DataFrame with outliers removed
    """
    df_clean = df.copy()
    
    for column in columns:
        if column in df_clean.columns and df_clean[column].dtype in ['int64', 'float64']:
            Q1 = df_clean[column].quantile(0.25)
            Q3 = df_clean[column].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            
            # Keep only rows where the column value is within bounds
            df_clean = df_clean[(df_clean[column] >= lower_bound) & (df_clean[column] <= upper_bound)]
    
    return df_clean

def remove_outliers_by_threshold(df, column, condition, threshold_value=None, min_threshold=None, max_threshold=None):
    """
    Remove outliers based on threshold conditions for a single column.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column name to apply threshold on
        condition (str): Condition type - "greater than", "less than", or "outside range"
        threshold_value (float): Single threshold value for greater/less than conditions
        min_threshold (float): Minimum threshold for range condition
        max_threshold (float): Maximum threshold for range condition
        
    Returns:
        pd.DataFrame: DataFrame with outliers removed based on threshold
    """
    df_clean = df.copy()
    
    if column not in df_clean.columns:
        return df_clean
        
    if condition == "greater than" and threshold_value is not None:
        # Remove rows where column value is greater than threshold
        df_clean = df_clean[df_clean[column] <= threshold_value]
    elif condition == "less than" and threshold_value is not None:
        # Remove rows where column value is less than threshold  
        df_clean = df_clean[df_clean[column] >= threshold_value]
    elif condition == "outside range" and min_threshold is not None and max_threshold is not None:
        # Keep only rows where column value is within the range
        df_clean = df_clean[(df_clean[column] >= min_threshold) & (df_clean[column] <= max_threshold)]
    
    return df_clean

# Logging Functions
class ModelLogger:
    def __init__(self, log_file='model_iterations.csv'):
        self.log_file = log_file
        self.logs = []
        
    def log_iteration(self, iteration_data):
        """
        Log a model iteration with preprocessing choices and performance metrics.
        
        Args:
            iteration_data (dict): Dictionary containing iteration information
        """
        # Flatten the metrics dictionary
        if 'metrics' in iteration_data:
            metrics = iteration_data.pop('metrics')
            iteration_data.update(metrics)
            
        # Add timestamp
        iteration_data['timestamp'] = pd.Timestamp.now().isoformat()
        
        self.logs.append(iteration_data)
        
    def save_logs(self):
        """Save logs to a CSV file."""
        df = pd.DataFrame(self.logs)
        # Ensure timestamp is the first column
        cols = ['timestamp'] + [col for col in df.columns if col != 'timestamp']
        df = df[cols]
        df.to_csv(self.log_file, index=False)
            
    def get_logs_df(self):
        """Get logs as a pandas DataFrame."""
        return pd.DataFrame(self.logs)
