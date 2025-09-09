"""
Optimized Functions Module for Streamlit ML Framework
Refactored with proper caching, state management, and error handling
"""

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
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
from typing import Dict, List, Optional, Tuple, Any
import time

# Import our new modules
from cache_manager import (
    cache_expensive_computation, 
    cache_model_objects, 
    performance_monitor,
    create_data_hash,
    PerformanceMonitor
)
from state_manager import DataFrameStateManager


# ============================================================================
# OPTIMIZED DATA PROCESSING FUNCTIONS
# ============================================================================

@performance_monitor("pca_computation")
def perform_pca_optimized(state_manager: DataFrameStateManager, 
                         variance_percentage: float) -> Tuple[bool, Dict[str, Any]]:
    """
    Optimized PCA computation with proper state management.
    
    Args:
        state_manager: The centralized state manager
        variance_percentage: Desired variance to retain
    
    Returns:
        Tuple of (success, results_dict)
    """
    try:
        data = state_manager.current_dataframe
        if data is None:
            return False, {"error": "No data available"}
        
        target_column = state_manager.get_target_column()
        categorical_columns = state_manager.get_categorical_columns()
        
        if not target_column:
            return False, {"error": "No target column specified"}
        
        # Create data hash for caching
        data_hash = create_data_hash(data)
        
        # Use cached computation if available
        result = _perform_pca_cached(
            data_hash, 
            data, 
            target_column, 
            categorical_columns, 
            variance_percentage
        )
        
        # Update state with results
        success = state_manager.update_dataframe(
            result['data'],
            operation='pca',
            description=f'Applied PCA retaining {variance_percentage}% variance',
            metadata={
                'variance_percentage': variance_percentage,
                'components_before': result['total_cols_before'],
                'components_after': result['total_cols_after'],
                'cumulative_variance': result['cum_var'].tolist()
            }
        )
        
        return success, result
        
    except Exception as e:
        st.error(f"Error in PCA computation: {str(e)}")
        return False, {"error": str(e)}


@cache_expensive_computation(ttl=1800)  # Cache for 30 minutes
def _perform_pca_cached(data_hash: str, data: pd.DataFrame, target_column: str, 
                       categorical_columns: List[str], variance_percentage: float) -> Dict[str, Any]:
    """Internal cached PCA implementation"""
    # Separate features and target
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Identify numerical columns
    numerical_columns = [col for col in X.columns if col not in categorical_columns]

    # Standardize the numerical columns
    X_scaled = X.copy()
    X_scaled[numerical_columns] = StandardScaler().fit_transform(X[numerical_columns])

    # Perform PCA
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled[numerical_columns])

    # Create DataFrame with principal components
    n_components = pca.n_components_
    pc_col_names = [f"PC_{i+1}" for i in range(n_components)]
    df_pca = pd.DataFrame(data=X_pca, columns=pc_col_names)

    # Calculate cumulative variance
    cum_var = np.cumsum(pca.explained_variance_ratio_)

    # Determine components to keep
    n_components_to_keep = np.argmax(cum_var >= variance_percentage / 100) + 1

    # Keep selected components
    df_pca_reduced = df_pca.iloc[:, :n_components_to_keep]

    # Add back categorical columns and target
    df_final = pd.concat([df_pca_reduced, data[categorical_columns], y], axis=1)

    return {
        'data': df_final,
        'total_cols_before': X.shape[1] + len(categorical_columns),
        'total_cols_after': df_final.shape[1] - 1,
        'cum_var': cum_var,
        'n_components_kept': n_components_to_keep
    }


@performance_monitor("tsne_computation")
def perform_tsne_optimized(state_manager: DataFrameStateManager, 
                          n_components: int = 2) -> Tuple[bool, Dict[str, Any]]:
    """
    Optimized t-SNE computation with state management.
    """
    try:
        data = state_manager.current_dataframe
        if data is None:
            return False, {"error": "No data available"}
        
        target_column = state_manager.get_target_column()
        categorical_columns = state_manager.get_categorical_columns()
        
        result = _perform_tsne_cached(
            create_data_hash(data),
            data,
            target_column,
            categorical_columns,
            n_components
        )
        
        success = state_manager.update_dataframe(
            result['data'],
            operation='tsne',
            description=f'Applied t-SNE with {n_components} components',
            metadata={'n_components': n_components}
        )
        
        return success, result
        
    except Exception as e:
        st.error(f"Error in t-SNE computation: {str(e)}")
        return False, {"error": str(e)}


@cache_expensive_computation(ttl=1800)
def _perform_tsne_cached(data_hash: str, data: pd.DataFrame, target_column: str,
                        categorical_columns: List[str], n_components: int) -> Dict[str, Any]:
    """Internal cached t-SNE implementation"""
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    numerical_columns = [col for col in X.columns if col not in categorical_columns]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X[numerical_columns])
    
    tsne = TSNE(n_components=n_components, random_state=42)
    X_tsne = tsne.fit_transform(X_scaled)
    
    df_tsne = pd.DataFrame(X_tsne, columns=[f"tSNE_{i+1}" for i in range(n_components)])
    df_final = pd.concat([df_tsne, data[categorical_columns], y], axis=1)
    
    return {
        'data': df_final,
        'n_features_before': len(numerical_columns),
        'n_features_after': n_components
    }


@performance_monitor("missing_value_imputation")
def impute_missing_values_optimized(state_manager: DataFrameStateManager,
                                   method: str = 'mean', 
                                   k: int = 5) -> bool:
    """
    Optimized missing value imputation with state management.
    """
    try:
        data = state_manager.current_dataframe
        if data is None:
            st.error("No data available for imputation")
            return False
        
        # Count missing values before
        missing_before = data.isna().sum().sum()
        
        # Perform imputation
        if method == 'mean':
            imputed_data = data.fillna(data.mean())
        elif method == 'median':
            imputed_data = data.fillna(data.median())
        elif method == 'mode':
            imputed_data = data.fillna(data.mode().iloc[0])
        elif method == 'knn':
            from sklearn.impute import KNNImputer
            imputer = KNNImputer(n_neighbors=k)
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            imputed_numeric = pd.DataFrame(
                imputer.fit_transform(data[numeric_columns]), 
                columns=numeric_columns,
                index=data.index
            )
            imputed_data = data.copy()
            imputed_data[numeric_columns] = imputed_numeric
        elif method == 'drop_rows':
            imputed_data = data.dropna()
        elif method == 'drop_cols':
            imputed_data = data.dropna(axis=1)
        else:
            st.error(f"Unknown imputation method: {method}")
            return False
        
        # Count missing values after
        missing_after = imputed_data.isna().sum().sum()
        
        # Update state
        success = state_manager.update_dataframe(
            imputed_data,
            operation=f'impute_{method}',
            description=f'Imputed missing values using {method}',
            metadata={
                'method': method,
                'missing_before': int(missing_before),
                'missing_after': int(missing_after),
                'k_neighbors': k if method == 'knn' else None
            }
        )
        
        if success:
            st.success(f"✅ Imputation completed: {missing_before} → {missing_after} missing values")
        
        return success
        
    except Exception as e:
        st.error(f"Error during imputation: {str(e)}")
        return False


@performance_monitor("data_smoothing")
def smooth_data_optimized(state_manager: DataFrameStateManager,
                         method: str = 'moving_average',
                         **params) -> bool:
    """
    Optimized data smoothing with state management.
    """
    try:
        data = state_manager.current_dataframe
        if data is None:
            st.error("No data available for smoothing")
            return False
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        result = data.copy()
        
        if method == 'moving_average':
            window_size = params.get('window_size', 3)
            result[numeric_cols] = result[numeric_cols].rolling(
                window=window_size, center=True
            ).mean()
        elif method == 'exponential':
            alpha = params.get('alpha', 0.3)
            result[numeric_cols] = result[numeric_cols].ewm(alpha=alpha).mean()
        elif method == 'savgol':
            from scipy.signal import savgol_filter
            window_size = params.get('window_size', 5)
            for col in numeric_cols:
                result[col] = savgol_filter(data[col], window_size, 3)
        elif method == 'lowess':
            from statsmodels.nonparametric.smoothers_lowess import lowess
            window_size = params.get('window_size', 0.3)
            for col in numeric_cols:
                result[col] = lowess(
                    data[col], np.arange(len(data[col])), 
                    frac=window_size
                )[:, 1]
        else:
            st.error(f"Unknown smoothing method: {method}")
            return False
        
        # Remove NaN values that might be introduced by smoothing
        result = result.dropna()
        
        success = state_manager.update_dataframe(
            result,
            operation=f'smooth_{method}',
            description=f'Applied {method} smoothing',
            metadata={'method': method, **params}
        )
        
        if success:
            st.success(f"✅ Smoothing applied using {method}")
        
        return success
        
    except Exception as e:
        st.error(f"Error during smoothing: {str(e)}")
        return False


# ============================================================================
# MODEL EVALUATION WITH OPTIMIZATION
# ============================================================================

@cache_model_objects()
def get_lazy_regressor():
    """Get cached LazyRegressor instance"""
    import LazyRegressor
    return LazyRegressor.LazyRegressor(verbose=1, ignore_warnings=False, custom_metric=None, predictions=True)


@performance_monitor("model_evaluation")
def evaluate_regression_models_optimized(X_train, X_test, y_train, y_test) -> pd.DataFrame:
    """
    Optimized model evaluation using cached regressor instance.
    """
    try:
        # Get cached regressor instance
        reg = get_lazy_regressor()
        
        # Fit and evaluate models
        models_df = reg.fit(X_train, X_test, y_train, y_test)
        
        return models_df
        
    except Exception as e:
        st.error(f"Error during model evaluation: {str(e)}")
        return pd.DataFrame()


# ============================================================================
# FEATURE ENGINEERING OPTIMIZATION
# ============================================================================

@performance_monitor("ndsi_calculation")
def calculate_ndsi_optimized(state_manager: DataFrameStateManager, 
                            threshold: float = 0.4, 
                            max_distance: int = 10) -> bool:
    """
    Optimized NDSI calculation with state management.
    """
    try:
        data = state_manager.current_dataframe
        if data is None:
            st.error("No data available")
            return False
        
        target_column = state_manager.get_target_column()
        categorical_columns = state_manager.get_categorical_columns()
        
        # Calculate NDSI correlations
        df_results = _ndsi_pearson_cached(
            create_data_hash(data),
            data,
            categorical_columns,
            target_column
        )
        
        # Display results and get top bands
        top_bands_list = _display_ndsi_heatmap_cached(
            create_data_hash(df_results),
            df_results,
            threshold,
            max_distance
        )
        
        # Calculate final NDSI
        final_ndsi_df = _calculate_ndsi_values_cached(
            create_data_hash(data),
            data,
            top_bands_list
        )
        
        # Combine with categorical and target columns
        result_df = pd.concat([
            final_ndsi_df, 
            data[categorical_columns], 
            data[target_column]
        ], axis=1)
        
        success = state_manager.update_dataframe(
            result_df,
            operation='ndsi_calculation',
            description=f'Calculated NDSI with threshold={threshold}',
            metadata={
                'threshold': threshold,
                'max_distance': max_distance,
                'n_bands_selected': len(top_bands_list),
                'original_features': len(data.columns) - len(categorical_columns) - 1,
                'final_features': len(final_ndsi_df.columns)
            }
        )
        
        if success:
            st.success(f"✅ NDSI calculation completed: {len(top_bands_list)} band pairs selected")
        
        return success
        
    except Exception as e:
        st.error(f"Error in NDSI calculation: {str(e)}")
        return False


@cache_expensive_computation(ttl=1800)
def _ndsi_pearson_cached(data_hash: str, data: pd.DataFrame, 
                        categorical_columns: List[str], target_col: str) -> pd.DataFrame:
    """Cached NDSI Pearson correlation calculation"""
    # Extract labels
    y = data[target_col].values
    df = data.drop(target_col, axis=1).drop(categorical_columns, axis=1)
    df.columns = df.columns.map(str)
    bands_list = df.columns

    # All possible pairs
    all_pairs = list(itertools.combinations(bands_list, 2))

    # Initialize arrays
    corrs = np.zeros(len(all_pairs))
    pvals = np.zeros(len(all_pairs))

    # Calculate NDSI and correlations
    for index, pair in enumerate(all_pairs):
        a = df[pair[0]].values
        b = df[pair[1]].values
        norm_index = (a - b) / (a + b)
        corr, pval = stats.pearsonr(norm_index, y)
        corrs[index] = corr
        pvals[index] = pval

    # Create results DataFrame
    col1 = [tple[0] for tple in all_pairs]
    col2 = [tple[1] for tple in all_pairs]
    index_col = [f"{tple[0]},{tple[1]}" for tple in all_pairs]
    
    data_dict = {
        'band1': col1, 
        "band2": col2, 
        'Pearson_Corr': corrs, 
        'p_value': pvals
    }
    
    df_results = pd.DataFrame(data=data_dict, index=index_col)
    df_results["Abs_Pearson_Corr"] = df_results["Pearson_Corr"].abs()
    
    return df_results.sort_values('Abs_Pearson_Corr', ascending=False)


@cache_expensive_computation(ttl=1800)
def _display_ndsi_heatmap_cached(results_hash: str, results: pd.DataFrame, 
                                threshold: float, max_distance: int) -> List[Tuple[str, str]]:
    """Cached NDSI heatmap display and top bands selection"""
    # Pivot for heatmap
    data = results.pivot(index='band1', columns='band2', values='Pearson_Corr')

    # Find local maxima and minima
    local_max = (maximum_filter(data, footprint=np.ones((max_distance, max_distance))) == data) & (data > threshold)
    local_min = (minimum_filter(data, footprint=np.ones((max_distance, max_distance))) == data) & (data < -threshold)

    # Get indices
    maxima_x, maxima_y = np.where(local_max)
    minima_x, minima_y = np.where(local_min)

    # Create band lists
    minima_list = [(data.index[minima_x][i], data.columns[minima_y][i]) for i in range(len(minima_x))]
    maxima_list = [(data.index[maxima_x][i], data.columns[maxima_y][i]) for i in range(len(maxima_x))]

    # Combine lists
    top_bands_list = minima_list + maxima_list

    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=data.values,
        x=data.columns,
        y=data.index,
        colorscale='RdBu',
        zmin=-1, zmax=1,
        colorbar=dict(title='Pearson Correlation')
    ))
    
    # Add markers for extrema
    fig.add_trace(go.Scatter(
        x=data.columns[maxima_y], 
        y=data.index[maxima_x], 
        mode='markers', 
        marker=dict(color='blue'), 
        name='Local Maxima'
    ))

    fig.add_trace(go.Scatter(
        x=data.columns[minima_y], 
        y=data.index[minima_x], 
        mode='markers', 
        marker=dict(color='red'), 
        name='Local Minima'
    ))

    fig.update_layout(
        title='NDSI Correlation Heatmap',
        xaxis_title='Band 2',
        yaxis_title='Band 1',
        height=600,
        width=800
    )

    st.plotly_chart(fig)

    return top_bands_list


@cache_expensive_computation(ttl=1800)
def _calculate_ndsi_values_cached(data_hash: str, data: pd.DataFrame, 
                                 top_bands_list: List[Tuple[str, str]]) -> pd.DataFrame:
    """Cached NDSI values calculation"""
    ndsi_df = pd.DataFrame()

    for tup in top_bands_list:
        a = data[tup[0]]
        b = data[tup[1]]
        ndsi = (a - b) / (a + b)
        column_name = f"{tup[0]}-{tup[1]}"
        ndsi_df[column_name] = ndsi

    return ndsi_df


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def display_processing_summary(state_manager: DataFrameStateManager):
    """Display a comprehensive processing summary"""
    status = state_manager.get_current_status()
    
    if not status['has_data']:
        st.warning("No data loaded for processing summary.")
        return
    
    st.subheader("📋 Processing Summary")
    
    # Basic info
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Original Shape", 
                 f"{status['original_shape'][0]} × {status['original_shape'][1]}" if status['original_shape'] else "N/A")
    
    with col2:
        st.metric("Current Shape", 
                 f"{status['current_shape'][0]} × {status['current_shape'][1]}" if status['current_shape'] else "N/A")
    
    with col3:
        st.metric("Processing Steps", status['processing_steps_count'])
    
    # Feature information
    features = state_manager.get_features()
    target = state_manager.get_target_column()
    categorical = state_manager.get_categorical_columns()
    
    if features or target:
        st.subheader("🎯 Feature Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Features:**")
            if features:
                st.write(f"- {len(features)} feature columns selected")
                with st.expander("View feature list"):
                    st.write(features)
            else:
                st.write("- No features selected")
        
        with col2:
            st.write("**Target & Categorical:**")
            st.write(f"- Target: {target if target else 'Not set'}")
            st.write(f"- Categorical: {len(categorical)} columns")
            if categorical:
                with st.expander("View categorical columns"):
                    st.write(categorical)


def create_operation_timeline(state_manager: DataFrameStateManager):
    """Create a visual timeline of all operations performed"""
    history = state_manager.get_operation_history()
    
    if not history:
        st.info("No operations performed yet.")
        return
    
    st.subheader("⏱️ Operation Timeline")
    
    timeline_data = []
    for i, op in enumerate(history):
        timeline_data.append({
            'Step': i + 1,
            'Operation': op['operation'],
            'Description': op.get('description', 'N/A'),
            'Timestamp': pd.to_datetime(op['timestamp']).strftime('%H:%M:%S'),
            'Current': i == len(history) - 1
        })
    
    df_timeline = pd.DataFrame(timeline_data)
    
    # Style the current operation
    def highlight_current(row):
        return ['background-color: lightgreen' if row['Current'] else '' for _ in row]
    
    styled_df = df_timeline.style.apply(highlight_current, axis=1)
    st.dataframe(styled_df, use_container_width=True)
