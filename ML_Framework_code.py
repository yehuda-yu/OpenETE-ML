import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
import shap
import pickle
import functions  # Assuming this contains your custom functions
import numbers

# Helper function to robustly replace values with NaN in all columns
def robust_replace_with_nan(df, values):
    df = df.copy()
    for v in values:
        # Replace as is
        df = df.replace(v, np.nan)
        # Replace as float (if possible)
        try:
            v_num = float(v)
            df = df.replace(v_num, np.nan)
        except:
            pass
        # Replace as string
        df = df.replace(str(v), np.nan)
    return df

# Configure Streamlit page settings
st.set_page_config(
    page_title="OpenETE-ML Framework",
    layout="wide",
    initial_sidebar_state="expanded"  # This makes the sidebar open by default
)

# Educational information dictionary for ML methods
ML_METHODS_INFO = {
    # Missing Value Handling
    "Mean Imputation": {
        "description": "Replaces missing values with the average of the column. Simple but may not preserve data distribution.",
        "link": "https://en.wikipedia.org/wiki/Imputation_(statistics)"
    },
    "Median Imputation": {
        "description": "Replaces missing values with the middle value of the column. More robust to outliers than mean.",
        "link": "https://en.wikipedia.org/wiki/Median"
    },
    "Mode Imputation": {
        "description": "Replaces missing values with the most frequent value. Best for categorical data.",
        "link": "https://en.wikipedia.org/wiki/Mode_(statistics)"
    },
    "KNN Imputation": {
        "description": "Replaces missing values using values from similar samples. Preserves data relationships but computationally expensive.",
        "link": "https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm"
    },
    "Drop Rows": {
        "description": "Removes entire rows with missing values. Simple but may lose valuable data.",
        "link": "https://en.wikipedia.org/wiki/Data_cleansing"
    },
    "Drop Columns": {
        "description": "Removes entire columns with missing values. Use when a column has too many missing values.",
        "link": "https://en.wikipedia.org/wiki/Data_cleansing"
    },
    
    # Data Smoothing
    "Moving Average": {
        "description": "Calculates the average of values within a sliding window. Reduces noise while preserving trends.",
        "link": "https://en.wikipedia.org/wiki/Moving_average"
    },
    "Exponential Smoothing": {
        "description": "Applies weighted averages favoring recent observations. Good for time series data with trends.",
        "link": "https://en.wikipedia.org/wiki/Exponential_smoothing"
    },
    "Savitzky-Golay": {
        "description": "Fits a polynomial to data points within a window. Preserves peak shapes better than simple averaging.",
        "link": "https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter"
    },
    "LOWESS": {
        "description": "Local regression smoothing that fits polynomials to subsets of data. Non-parametric and robust to outliers.",
        "link": "https://en.wikipedia.org/wiki/Local_regression"
    },
    
    # Outlier Handling
    "IQR Method (on one or more columns)": {
        "description": "Removes outliers using the Interquartile Range method. Values beyond Q1-1.5*IQR or Q3+1.5*IQR are considered outliers.",
        "link": "https://en.wikipedia.org/wiki/Outlier"
    },
    "Threshold Method (on a single column)": {
        "description": "Removes outliers based on custom threshold values. Allows filtering by greater than, less than, or outside a specified range.",
        "link": "https://en.wikipedia.org/wiki/Outlier"
    },
    
    # Data Normalization
    "MinMaxScaler": {
        "description": "Scales data to a fixed range (usually 0-1). Preserves zero entries and sparse data structure.",
        "link": "https://en.wikipedia.org/wiki/Feature_scaling"
    },
    "StandardScaler": {
        "description": "Transforms data to have zero mean and unit variance. Assumes data follows normal distribution.",
        "link": "https://en.wikipedia.org/wiki/Standard_score"
    },
    
    # Categorical Encoding
    "OneHotEncoder": {
        "description": "Creates binary columns for each category. Preserves all information but increases dimensionality.",
        "link": "https://en.wikipedia.org/wiki/One-hot"
    },
    "LabelEncoder": {
        "description": "Converts categories to numerical labels. Simple but assumes ordinal relationships between categories.",
        "link": "https://en.wikipedia.org/wiki/Categorical_variable"
    },
    
    # Feature Extraction
    "PCA": {
        "description": "Reduces dimensions by finding directions of maximum variance. Preserves most important information while reducing complexity.",
        "link": "https://en.wikipedia.org/wiki/Principal_component_analysis"
    },
    "Time Series": {
        "description": "Creates features from temporal patterns like lag values, rolling statistics, and seasonal components.",
        "link": "https://en.wikipedia.org/wiki/Time_series"
    },
    "t-SNE": {
        "description": "Non-linear dimensionality reduction for visualization. Preserves local structure of high-dimensional data.",
        "link": "https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding"
    },
    
    # Feature Selection
    "NDSI": {
        "description": "Calculates spectral indices from band combinations. Commonly used in remote sensing for vegetation analysis.",
        "link": "https://www.sciencedirect.com/science/article/pii/S003442570700185X"
    },
    "SelectKBest": {
        "description": "Selects top k features based on statistical tests (F-test, mutual information, etc.). Fast and effective for initial feature screening.",
        "link": "https://en.wikipedia.org/wiki/Feature_selection"
    },
    "RFE": {
        "description": "Iteratively removes least important features. Uses model performance to determine feature importance.",
        "link": "https://en.wikipedia.org/wiki/Feature_selection"
    },
    
    # Cross-Validation
    "Cross-Validation": {
        "description": "A technique to assess model performance by splitting data into k folds. Each fold serves as test data while others train the model, providing robust performance estimates.",
        "link": "https://en.wikipedia.org/wiki/Cross-validation_(statistics)"
    },
    
    # Models (key ones)
    "Linear Regression": {
        "description": "Fits a straight line to data. Simple, interpretable, and fast. Assumes linear relationship between features and target.",
        "link": "https://en.wikipedia.org/wiki/Linear_regression"
    },
    "Random Forest": {
        "description": "Ensemble of decision trees with random feature subsets. Reduces overfitting and provides feature importance.",
        "link": "https://en.wikipedia.org/wiki/Random_forest"
    },
    "XGBoost": {
        "description": "Optimized gradient boosting implementation. Often achieves state-of-the-art performance on structured data.",
        "link": "https://en.wikipedia.org/wiki/XGBoost"
    },
    "SVR": {
        "description": "Uses support vectors to find optimal hyperplane for regression. Effective for non-linear relationships with kernel functions.",
        "link": "https://en.wikipedia.org/wiki/Support_vector_machine"
    }
}

def show_method_info(method_name, show_educational_info):
    """Display educational information about a method if enabled"""
    if show_educational_info and method_name in ML_METHODS_INFO:
        info = ML_METHODS_INFO[method_name]
        with st.expander(f"📚 Learn about {method_name}", expanded=False):
            st.write(f"**{method_name}**: {info['description']}")
            st.markdown(f"[Learn more on Wikipedia]({info['link']})")

# Helper function to create a tooltip
def tooltip(text):
    return st.markdown(f"""<span title="{text}">❔</span>""", unsafe_allow_html=True)

# Use session_state to store actions and parameters
if 'state' not in st.session_state:
    st.session_state.state = {}

if 'models_evaluated' not in st.session_state:
    st.session_state.models_evaluated = False
if 'best_model' not in st.session_state:
    st.session_state.best_model = None
if 'current_step' not in st.session_state:
    st.session_state.current_step = 1

# Initialize the model logger
if 'model_logger' not in st.session_state:
    st.session_state.model_logger = functions.ModelLogger()

# Educational info toggle in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### 🎓 Educational Mode")
st.sidebar.markdown("""
**Need explanations of ML concepts and methods?**

Turn on educational mode""")
show_educational_info = st.sidebar.checkbox(
    "Show method explanations", 
    value=False,
    help="Enable to see explanations and links for ML methods"
)

# Sidebar for step navigation
st.sidebar.title("📊 ML Workflow Progress")

# Define steps with descriptions
steps = {
    1: {"title": "Upload Data", "description": "Upload and explore your dataset"},
    2: {"title": "Feature Selection", "description": "Select features and target variable"},
    3: {"title": "Data Processing", "description": "Clean and preprocess your data"},
    4: {"title": "Feature Engineering", "description": "Reduce features and extract new ones"},
    5: {"title": "Model Training", "description": "Train and evaluate models"}
}

# --- Workflow Progress: Dynamically update current_step based on explicit step flags ---
def get_current_step():
    if not st.session_state.get('step1_done', False):
        return 1
    elif not st.session_state.get('step2_done', False):
        return 2
    elif not st.session_state.get('step3_done', False):
        return 3
    elif (not st.session_state.get('step4_done', False)) and st.session_state.get('step5_done', False):
        return 5
    elif not st.session_state.get('step4_done', False):
        return 4
    elif not st.session_state.get('step5_done', False):
        return 5
    else:
        return 5

# Call this at the top of the main logic to keep sidebar in sync
st.session_state.current_step = get_current_step()

# Display steps in sidebar
for step_num, step_info in steps.items():
    # Determine if step is completed, current, or upcoming
    is_completed = step_num < st.session_state.current_step
    is_current = step_num == st.session_state.current_step
    
    # Create step display
    if is_completed:
        st.sidebar.markdown(f"""
        <div style="padding: 10px; margin: 5px 0; background-color: #d4edda; border-radius: 5px; border-left: 4px solid #28a745;">
            <div style="display: flex; align-items: center;">
                <span style="font-size: 20px; margin-right: 10px;">✅</span>
                <div>
                    <strong>Step {step_num}: {step_info['title']}</strong><br>
                    <small style="color: #6c757d;">{step_info['description']}</small>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    elif is_current:
        st.sidebar.markdown(f"""
        <div style="padding: 10px; margin: 5px 0; background-color: #fff3cd; border-radius: 5px; border-left: 4px solid #ffc107;">
            <div style="display: flex; align-items: center;">
                <span style="font-size: 20px; margin-right: 10px;">🔄</span>
                <div>
                    <strong>Step {step_num}: {step_info['title']}</strong><br>
                    <small style="color: #856404;">{step_info['description']}</small>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.sidebar.markdown(f"""
        <div style="padding: 10px; margin: 5px 0; background-color: #f8f9fa; border-radius: 5px; border-left: 4px solid #6c757d;">
            <div style="display: flex; align-items: center;">
                <span style="font-size: 20px; margin-right: 10px;">⏳</span>
                <div>
                    <strong style="color: #6c757d;">Step {step_num}: {step_info['title']}</strong><br>
                    <small style="color: #6c757d;">{step_info['description']}</small>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# Progress bar
progress_percentage = (st.session_state.current_step - 1) / len(steps) * 100
st.sidebar.progress(progress_percentage / 100)
st.sidebar.markdown(f"**Progress: {progress_percentage:.0f}%**")

# Add some spacing
st.sidebar.markdown("---")

# Quick navigation info
st.sidebar.markdown("""
### 💡 Tips
- Complete each step to proceed
- Use the tooltips (❔) for help
- Your progress is automatically saved
""")

# Create a title for the app
st.title("OpenETE-ML End to End ML Regression Model Builder")

# Step 1: Upload Data
st.header("Step 1: Upload Data")
col1, col2 = st.columns([10, 1])
with col1:
    uploaded_file = st.file_uploader("Upload your data file", type=["csv", "xlsx"])
with col2:
    tooltip("Upload a CSV or Excel file containing your dataset for analysis")

# This block handles the data loading and state management.
# It ensures data is loaded only once per file and all subsequent operations
# use the persistent DataFrame in st.session_state.
if uploaded_file is not None:
    # Check if a new file has been uploaded.
    is_new_file = 'uploaded_file_name' not in st.session_state or st.session_state.uploaded_file_name != uploaded_file.name
    
    if is_new_file:
        # If it's a new file, reset the entire session state and load the new data.
        for key in list(st.session_state.keys()):
            if not key.startswith('_'):  # Avoid touching Streamlit's internal keys
                del st.session_state[key]
        
        try:
            # Load data and set initial state
            data = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
            st.session_state.data = data.copy()
            st.session_state.uploaded_file_name = uploaded_file.name
            st.session_state.step1_done = True
        except Exception as e:
            st.error(f"Error reading file: {e}")
            st.stop()

    # --- From here, the entire app logic runs using the persistent st.session_state.data ---
    st.subheader("Full Dataset")
    st.dataframe(st.session_state.data)
    
    # Data Overview
    st.write("### Data Overview")
    st.write(f"Number of samples: {st.session_state.data.shape[0]}")
    st.write(f"Number of columns: {st.session_state.data.shape[1]}")

    # NaN values
    nan_counts = st.session_state.data.isna().sum()
    total_nans = nan_counts.sum()
    st.write(f"Total NaN values: {total_nans}")
    if total_nans > 0:
        with st.expander("NaN counts per column"):
            st.write(nan_counts)
    else:
        st.success("No NaN values in the dataset! ✅")
    
    # Option 1: Remove rows from the beginning
    col1, col2 = st.columns([10, 1])
    with col1:
        remove_rows = st.checkbox("Remove rows from the beginning of the dataset")
    with col2:
        tooltip("Remove the first N rows from your dataset (useful for removing headers or metadata)")
    
    if remove_rows:
        col1, col2 = st.columns([10, 1])
        with col1:
            n_rows_to_remove = st.number_input(
                "Number of rows to remove from the beginning", 
                min_value=1, 
                max_value=len(st.session_state.data)-1, 
                value=1, 
                step=1
            )
        with col2:
            tooltip(f"Remove the first {n_rows_to_remove} rows from your dataset")
        
        if st.button("Apply Row Removal"):
            st.session_state.data = st.session_state.data.iloc[n_rows_to_remove:].reset_index(drop=True)
            st.success(f"Removed first {n_rows_to_remove} rows. New dataset size: {len(st.session_state.data)} rows")
            with st.expander("View Updated Dataset After Row Removal"):
                st.dataframe(st.session_state.data)

    # Feature columns selection
    st.header("Step 2: Feature Columns Selection")
    col1, col2 = st.columns([10, 1])
    with col1:
        # If features are already selected and stored in session state, use them as default
        # But ensure they exist in current data columns to avoid the error
        stored_features = st.session_state.get('features', [])
        current_columns = st.session_state.data.columns.tolist()
        
        # Filter stored features to only include those that exist in current data
        valid_default_features = [f for f in stored_features if f in current_columns]
        
        # If no valid stored features, use all columns as default
        if not valid_default_features:
            valid_default_features = current_columns
        
        features = st.multiselect(
            "Select features columns", 
            current_columns, 
            default=valid_default_features
        )
    with col2:
        tooltip("Select the columns from your dataset that will be used as features (independent variables) for the model")

    if features:
        st.session_state.step2_done = True

    # Categorical columns
    col1, col2 = st.columns([10, 1])
    with col1:
        # Use a key to persist the state of this checkbox
        has_categorical_columns = st.checkbox("Are there categorical columns?", key='has_categorical_columns')
    with col2:
        tooltip("Check this if your dataset contains categorical variables (non-numeric data like text or categories)")

    if has_categorical_columns:
        col1, col2 = st.columns([10, 1])
        with col1:
            # If categorical columns are already selected, use them as default
            # But ensure they exist in current features list to avoid the error
            stored_categorical = st.session_state.get('categorical_columns', [])
            valid_categorical = [c for c in stored_categorical if c in features]
            categorical_columns = st.multiselect("Select categorical columns", features, default=valid_categorical)
        with col2:
            tooltip("Select the columns that contain categorical data (text, labels, etc.)")
    else:
        categorical_columns = []

    # Target column selection
    st.header("Step 3: Target Column Selection")
    col1, col2 = st.columns([10, 1])
    with col1:
        target_options = [col for col in st.session_state.data.columns if col not in features]
        
        # Determine the index for the default value of the selectbox
        default_target_index = 0
        if 'target_column' in st.session_state and st.session_state.target_column in target_options:
            default_target_index = target_options.index(st.session_state.target_column)
        elif len(target_options) == 0:
            st.error("No columns available for target selection. Please ensure you have selected features that leave at least one column for the target.")
            st.stop()
        
        target_column = st.selectbox("Select the target column", target_options, index=default_target_index)
    with col2:
        tooltip("Select the column you want to predict (dependent variable)")

    # Add Apply button for feature + target selection
    apply_target = st.button("Apply Feature & Target Selection")
    if apply_target:
        # Store selections in session_state to persist them
        st.session_state.features = features
        st.session_state.categorical_columns = categorical_columns
        st.session_state.target_column = target_column
        
        st.session_state.data = st.session_state.data[features + [target_column]]
        with st.expander("View Data After Feature & Target Selection"):
            st.dataframe(st.session_state.data)

    # Data Processing Options
    st.header("Step 4: Data Processing Options")
    
    # --- Replace specific values with NaN (before NaN handling) ---
    col1, col2 = st.columns([10, 1])
    with col1:
        replace_values = st.checkbox("Replace specific values with NaN")
    with col2:
        tooltip("Replace specific values (like -999, 0, or other placeholders) with NaN")
    
    if replace_values:
        col1, col2 = st.columns([10, 1])
        with col1:
            replacement_method = st.radio(
                "How would you like to specify values to replace?",
                ["Single value", "Multiple values", "Range of values"]
            )
        with col2:
            tooltip("Choose how to specify which values should be replaced with NaN")
        
        if replacement_method == "Single value":
            col1, col2 = st.columns([10, 1])
            with col1:
                value_to_replace = st.text_input("Enter the value to replace with NaN (numeric or text)", value="-999")
            with col2:
                tooltip("All occurrences of this value (as string or number) will be replaced with NaN")
            
            if st.button("Apply Single Value Replacement"):
                st.session_state.data = st.session_state.data.replace(value_to_replace, np.nan)
                try:
                    v_num = float(value_to_replace)
                    st.session_state.data = st.session_state.data.replace(v_num, np.nan)
                except:
                    pass
                st.session_state.data = st.session_state.data.replace(str(value_to_replace), np.nan)
                st.success(f"Replaced all occurrences of '{value_to_replace}' with NaN.")
                nan_counts_updated = st.session_state.data.isna().sum()
                total_nans_updated = nan_counts_updated.sum()
                if total_nans_updated == 0:
                    st.success("No NaN values in the dataset! ✅")
                else:
                    st.write(f"Total NaN values after replacement: {total_nans_updated}")
                    with st.expander("NaN counts per column"):
                        st.write(nan_counts_updated)
                with st.expander("View Updated Data After Replacement"):
                    st.dataframe(st.session_state.data)
        
        elif replacement_method == "Multiple values":
            col1, col2 = st.columns([10, 1])
            with col1:
                values_input = st.text_input(
                    "Enter values to replace with NaN (comma-separated, numeric or text)", 
                    value="-999, -9999, 0"
                )
            with col2:
                tooltip("Enter multiple values separated by commas. All these values (as string or number) will be replaced with NaN")
            
            if st.button("Apply Multiple Values Replacement"):
                try:
                    values_to_replace = [v.strip() for v in values_input.split(',')]
                    for v in values_to_replace:
                        st.session_state.data = st.session_state.data.replace(v, np.nan)
                        try:
                            v_num = float(v)
                            st.session_state.data = st.session_state.data.replace(v_num, np.nan)
                        except:
                            pass
                        st.session_state.data = st.session_state.data.replace(str(v), np.nan)
                    st.success(f"Replaced all occurrences of {values_to_replace} with NaN.")
                    nan_counts_updated = st.session_state.data.isna().sum()
                    total_nans_updated = nan_counts_updated.sum()
                    if total_nans_updated == 0:
                        st.success("No NaN values in the dataset! ✅")
                    else:
                        st.write(f"Total NaN values after replacement: {total_nans_updated}")
                        with st.expander("NaN counts per column"):
                            st.write(nan_counts_updated)
                    with st.expander("View Updated Data After Replacement"):
                        st.dataframe(st.session_state.data)
                except Exception as e:
                    st.error(f"Error: {e}. Please enter valid values separated by commas.")
        
        elif replacement_method == "Range of values":
            col1, col2 = st.columns(2)
            with col1:
                min_val = st.number_input("Minimum value", value=-1000.0)
                tooltip("Values inside this range will be replaced with NaN")
            with col2:
                max_val = st.number_input("Maximum value", value=1000.0)
                tooltip("Values inside this range will be replaced with NaN")
            
            if st.button("Apply Range Replacement"):
                def replace_in_range_with_nan(df, min_val, max_val):
                    df_replaced = df.copy()
                    non_numeric_cols = []
                    replaced_count = 0
                    for col in df.columns:
                        col_numeric = pd.to_numeric(df[col], errors='coerce')
                        mask = col_numeric.notna() & (col_numeric >= min_val) & (col_numeric <= max_val)
                        replaced_count += mask.sum()
                        df_replaced.loc[mask, col] = np.nan
                        if col_numeric.notna().sum() == 0:
                            non_numeric_cols.append(col)
                    return df_replaced, replaced_count, non_numeric_cols
                st.session_state.data, replaced_count, non_numeric_cols = replace_in_range_with_nan(st.session_state.data, min_val, max_val)
                st.success(f"Replaced {replaced_count} values inside range [{min_val}, {max_val}] with NaN.")
                if non_numeric_cols:
                    st.info(f"The following columns could not be checked for range (non-numeric data): {', '.join(non_numeric_cols)}")
                nan_counts_updated = st.session_state.data.isna().sum()
                total_nans_updated = nan_counts_updated.sum()
                if total_nans_updated == 0:
                    st.success("No NaN values in the dataset! ✅")
                else:
                    st.write(f"Total NaN values after replacement: {total_nans_updated}")
                    with st.expander("NaN counts per column"):
                        st.write(nan_counts_updated)
                with st.expander("View Updated Data After Replacement"):
                    st.dataframe(st.session_state.data)


    # --- Outlier Handling (before missing values handling) ---
    col1, col2 = st.columns([10, 1])
    with col1:
        handle_outliers = st.checkbox("Handle Outliers")
    with col2:
        tooltip("Remove outliers that may negatively impact model performance")

    if handle_outliers:
        col1, col2 = st.columns([10, 1])
        with col1:
            outlier_method = st.radio(
                "Choose outlier removal method",
                ["IQR Method (on one or more columns)", "Threshold Method (on a single column)"]
            )
        with col2:
            tooltip("IQR Method: Uses statistical quartiles to identify outliers. Threshold Method: Uses custom thresholds to remove values")

        # Get numeric columns for outlier handling
        numeric_columns = st.session_state.data.select_dtypes(include=[np.number]).columns.tolist()
        
        if outlier_method == "IQR Method (on one or more columns)":
            col1, col2 = st.columns([10, 1])
            with col1:
                iqr_columns = st.multiselect(
                    "Select columns to check for outliers",
                    numeric_columns,
                    default=numeric_columns[:min(3, len(numeric_columns))]  # Default to first 3 numeric columns
                )
            with col2:
                tooltip("Select which numeric columns to apply IQR outlier detection on")
            
            col1, col2 = st.columns([10, 1])
            with col1:
                iqr_multiplier = st.number_input(
                    "IQR Multiplier",
                    min_value=1.0,
                    max_value=3.0,
                    value=1.5,
                    step=0.1,
                    help="Higher values are more permissive (keep more data)"
                )
            with col2:
                tooltip("Multiplier for IQR bounds. 1.5 is standard, higher values keep more data")
                
        elif outlier_method == "Threshold Method (on a single column)":
            col1, col2 = st.columns([10, 1])
            with col1:
                threshold_column = st.selectbox(
                    "Select column for threshold filtering",
                    numeric_columns
                )
            with col2:
                tooltip("Choose the numeric column to apply threshold filtering on")
            
            col1, col2 = st.columns([10, 1])
            with col1:
                condition = st.selectbox(
                    "Condition",
                    ["Remove values greater than", "Remove values less than", "Remove values outside range [min, max]"]
                )
            with col2:
                tooltip("Choose how to filter the data based on threshold values")
            
            if condition in ["Remove values greater than", "Remove values less than"]:
                col1, col2 = st.columns([10, 1])
                with col1:
                    threshold_value = st.number_input(
                        f"Threshold value",
                        value=float(st.session_state.data[threshold_column].median()) if threshold_column else 0.0
                    )
                with col2:
                    tooltip("Values will be filtered based on this threshold")
            elif condition == "Remove values outside range [min, max]":
                col1, col2, col3, col4 = st.columns([5, 1, 5, 1])
                with col1:
                    min_threshold = st.number_input(
                        "Minimum threshold",
                        value=float(st.session_state.data[threshold_column].quantile(0.05)) if threshold_column else 0.0
                    )
                with col2:
                    tooltip("Lower bound of acceptable range")
                with col3:
                    max_threshold = st.number_input(
                        "Maximum threshold", 
                        value=float(st.session_state.data[threshold_column].quantile(0.95)) if threshold_column else 1.0
                    )
                with col4:
                    tooltip("Upper bound of acceptable range")

        # Apply Outlier Removal button
        if st.button("Apply Outlier Removal"):
            original_rows = len(st.session_state.data)
            
            if outlier_method == "IQR Method (on one or more columns)":
                if iqr_columns:
                    st.session_state.data = functions.remove_outliers_iqr(
                        st.session_state.data, 
                        iqr_columns, 
                        iqr_multiplier
                    )
                else:
                    st.warning("Please select at least one column for IQR outlier removal.")
            
            elif outlier_method == "Threshold Method (on a single column)":
                if threshold_column:
                    if condition in ["Remove values greater than", "Remove values less than"]:
                        st.session_state.data = functions.remove_outliers_by_threshold(
                            st.session_state.data,
                            threshold_column,
                            condition.replace("Remove values ", ""),
                            threshold_value=threshold_value
                        )
                    elif condition == "Remove values outside range [min, max]":
                        st.session_state.data = functions.remove_outliers_by_threshold(
                            st.session_state.data,
                            threshold_column,
                            "outside range",
                            min_threshold=min_threshold,
                            max_threshold=max_threshold
                        )
                else:
                    st.warning("Please select a column for threshold outlier removal.")
            
            new_rows = len(st.session_state.data)
            rows_removed = original_rows - new_rows
            
            if rows_removed > 0:
                st.success(f"Outlier removal complete! Removed {rows_removed} rows ({rows_removed/original_rows*100:.1f}% of data). Dataset now has {new_rows} rows.")
            else:
                st.info("No outliers were detected and removed with the current settings.")


    # --- Missing Values Handling (after replacement) ---
    col1, col2 = st.columns([10, 1])
    with col1:
        handle_missing = st.checkbox("Handle missing values")
    with col2:
        tooltip("Enable this option to handle missing data (NaN values) in your dataset")

    if handle_missing:
        col1, col2 = st.columns([10, 1])
        with col1:
            missing_values_option = st.selectbox(
                "Choose missing values handling method",
                ["Mean Imputation", "Median Imputation", "Mode Imputation", "KNN Imputation", 
                 "Drop Rows", "Drop Columns"]
            )
        with col2:
            tooltip("Mean/Median/Mode: Replace missing values with averages. KNN: Replace with values from similar samples. Drop Rows/Columns: Remove data with missing values")
        show_method_info(missing_values_option, show_educational_info)
        if missing_values_option == "KNN Imputation":
            col1, col2 = st.columns([10, 1])
            with col1:
                k_neighbors = st.slider("Number of neighbors for KNN", 1, 10, 5)
            with col2:
                tooltip("Number of nearest neighbors to consider when imputing missing values")
        # --- Apply missing value handling with a button ---
        if st.button("Apply Missing Value Handling"):
            method_map = {
                "Mean Imputation": "mean",
                "Median Imputation": "median",
                "Mode Imputation": "mode",
                "KNN Imputation": "knn",
                "Drop Rows": "drop_rows",
                "Drop Columns": "drop_cols"
            }
            method = method_map[missing_values_option]
            k = k_neighbors if missing_values_option == "KNN Imputation" else 5
            st.session_state.data = functions.impute_missing_values(st.session_state.data, method=method, k=k)
            st.success(f"Applied {missing_values_option} to missing values.")
            with st.expander("View Data After Missing Value Handling"):
                st.dataframe(st.session_state.data)


    # --- Date Column and Resampling (now after NaN handling) ---
    col1, col2 = st.columns([10, 1])
    with col1:
        has_date_column = st.checkbox("Define date column for time-based resampling")
    with col2:
        tooltip("Enable this option if your dataset has a date/time column for time-based aggregation")
    if has_date_column:
        col1, col2 = st.columns([10, 1])
        with col1:
            date_column = st.selectbox("Select the date column", st.session_state.data.columns.tolist())
        with col2:
            tooltip("Select the column containing date/time information")
        
        col1, col2 = st.columns([10, 1])
        with col1:
            resample_frequency = st.selectbox(
                "Select resampling frequency",
                ["30T", "H", "D", "W", "M", "Q", "Y"],
                format_func=lambda x: {
                    "30T": "Half Hour",
                    "H": "Hour", 
                    "D": "Day",
                    "W": "Week",
                    "M": "Month",
                    "Q": "Quarter",
                    "Y": "Year"
                }[x]
            )
        with col2:
            tooltip("Select the time interval for resampling your data")
        col1, col2 = st.columns([10, 1])
        with col1:
            resample_method = st.selectbox(
                "Select resampling method",
                ["mean", "sum", "median", "min", "max", "std", "count"]
            )
        with col2:
            tooltip("Choose how to aggregate data within each time interval")
        if st.button("Apply Time Resampling"):
            try:
                # Smart auto-detection of date format - try multiple methods
                def smart_datetime_convert(series):
                    """Try multiple datetime parsing methods automatically"""
                    # Method 1: Default pandas parsing
                    try:
                        return pd.to_datetime(series, infer_datetime_format=True)
                    except:
                        pass
                    
                    # Method 2: Try with dayfirst=True (DD/MM/YYYY format)
                    try:
                        return pd.to_datetime(series, dayfirst=True)
                    except:
                        pass
                    
                    # Method 3: Try with dayfirst=False (MM/DD/YYYY format)
                    try:
                        return pd.to_datetime(series, dayfirst=False)
                    except:
                        pass
                    
                    # Method 4: Try common European format patterns
                    common_formats = [
                        '%d/%m/%Y %H:%M',
                        '%d/%m/%Y %H:%M:%S',
                        '%d-%m-%Y %H:%M',
                        '%d-%m-%Y %H:%M:%S',
                        '%d.%m.%Y %H:%M',
                        '%d.%m.%Y %H:%M:%S',
                        '%d/%m/%Y',
                        '%d-%m-%Y',
                        '%d.%m.%Y'
                    ]
                    
                    for fmt in common_formats:
                        try:
                            return pd.to_datetime(series, format=fmt)
                        except:
                            continue
                    
                    # Method 5: Try common US format patterns
                    us_formats = [
                        '%m/%d/%Y %H:%M',
                        '%m/%d/%Y %H:%M:%S',
                        '%m-%d-%Y %H:%M',
                        '%m-%d-%Y %H:%M:%S',
                        '%m/%d/%Y',
                        '%m-%d-%Y'
                    ]
                    
                    for fmt in us_formats:
                        try:
                            return pd.to_datetime(series, format=fmt)
                        except:
                            continue
                    
                    # Method 6: Try ISO format
                    try:
                        return pd.to_datetime(series, format='ISO8601')
                    except:
                        pass
                    
                    # Method 7: Last resort - mixed format
                    try:
                        return pd.to_datetime(series, format='mixed')
                    except:
                        pass
                    
                    # If all methods fail, raise an error
                    raise ValueError("Unable to parse datetime format automatically")

                # Convert date column to datetime format using smart detection
                st.session_state.data[date_column] = smart_datetime_convert(st.session_state.data[date_column])
                st.success(f"Successfully converted {date_column} to datetime format (auto-detected)")
                
                # Set date column as index for resampling
                st.session_state.data = st.session_state.data.set_index(date_column)
                
                # Apply resampling
                st.session_state.data = st.session_state.data.resample(resample_frequency).agg(resample_method)
                st.session_state.data = st.session_state.data.reset_index()
                st.success(f"Applied {resample_method} resampling with {resample_frequency} frequency")
                with st.expander("View Resampled Data"):
                    st.dataframe(st.session_state.data)
            except Exception as e:
                st.error(f"Error during time resampling: {e}")
                st.error("The system tried multiple date formats but couldn't parse your date column automatically.")
                
                # Show sample of problematic data for debugging
                with st.expander("Sample of date column data for debugging"):
                    st.write("First 10 values in the date column:")
                    st.write(st.session_state.data[date_column].head(10).tolist())
                    st.write("Please ensure your date column contains valid date/time values.")

    # Data Smoothing
    col1, col2 = st.columns([10, 1])
    with col1:
        apply_smoothing = st.checkbox("Apply data smoothing")
    with col2:
        tooltip("Smooth noisy data to improve model performance")

    if apply_smoothing:
        col1, col2 = st.columns([10, 1])
        with col1:
            smoothing_method = st.selectbox(
                "Choose smoothing method",
                ["Moving Average", "Exponential Smoothing", "Savitzky-Golay", "LOWESS"]
            )
        with col2:
            tooltip("Moving Average: Simple average over window. Exponential: Weighted average favoring recent data. Savitzky-Golay: Polynomial fit. LOWESS: Local regression")
        
        # Show educational info for smoothing methods
        show_method_info(smoothing_method, show_educational_info)
        
        if smoothing_method in ["Moving Average", "Savitzky-Golay"]:
            col1, col2 = st.columns([10, 1])
            with col1:
                window_size = st.slider("Window size", 3, 21, 5, step=2)
            with col2:
                tooltip("Size of the window for smoothing. Larger windows create smoother results but may lose detail")
        elif smoothing_method == "Exponential Smoothing":
            col1, col2 = st.columns([10, 1])
            with col1:
                alpha = st.slider("Smoothing factor (alpha)", 0.0, 1.0, 0.3, step=0.1)
            with col2:
                tooltip("Weight for current observation (0-1). Higher values give more weight to recent observations")
        elif smoothing_method == "LOWESS":
            col1, col2 = st.columns([10, 1])
            with col1:
                frac = st.slider("LOWESS window size (fraction)", 0.1, 1.0, 0.3, step=0.1)
            with col2:
                tooltip("Fraction of data to use for each local regression. Larger values produce smoother fits")

    # Normalization
    col1, col2 = st.columns([10, 1])
    with col1:
        normalize_data = st.checkbox("Normalize data")
    with col2:
        tooltip("Scale your data to a standard range. Important for many machine learning algorithms")

    if normalize_data:
        col1, col2 = st.columns([10, 1])
        with col1:
            normalization_method = st.radio("Choose normalization method", ["MinMaxScaler", "StandardScaler"])
        with col2:
            tooltip("MinMaxScaler: Scales to a range (0-1). StandardScaler: Transforms to mean=0, std=1")
        
        # Show educational info for normalization methods
        show_method_info(normalization_method, show_educational_info)

    # Categorical Variables
    col1, col2 = st.columns([10, 1])
    with col1:
        encode_categorical_variables = st.checkbox("Encode categorical variables")
    with col2:
        tooltip("Convert categorical data into numerical format for model training")

    if encode_categorical_variables and has_categorical_columns:
        col1, col2 = st.columns([10, 1])
        with col1:
            categorical_encoding_method = st.radio("Choose categorical encoding method", ["OneHotEncoder", "LabelEncoder"])
        with col2:
            tooltip("OneHotEncoder: Creates binary columns for each category. LabelEncoder: Converts categories to numerical labels")
        
        # Show educational info for encoding methods
        show_method_info(categorical_encoding_method, show_educational_info)


    # ...
    # At the end of all data processing options (after missing value handling, resampling, smoothing, normalization, encoding, etc.), set:
    # (You may want to set this after the user clicks a button to confirm data processing is done, or after a key action)
    # For now, set it after the last data processing block before feature engineering:
    st.session_state.step3_done = True

    # Feature Selection/Extraction
    st.header("Step 5: Feature Selection/Extraction Options")
    col1, col2 = st.columns([10, 1])
    with col1:
        reduce_features = st.checkbox("Enable feature reduction/selection", 
                                     help="Check this to configure feature reduction methods")
    with col2:
        tooltip("Enable this option to decrease the number of features in your dataset using dimensionality reduction techniques")
    
    # Add skip button for users who don't want feature selection
    if not reduce_features:
        if st.button("Skip Feature Selection and Continue", key="skip_feature_selection"):
            st.session_state.step4_done = True
            st.success("Feature selection skipped. Proceeding with all available features.")
    
    # Add reset button in case something goes wrong
    if st.button("Reset Feature Selection", key="reset_feature_selection", help="Click this if you encounter errors and want to reset"):
        # Clear any stored NDSI results or other temporary data
        if 'ndsi_results' in st.session_state:
            del st.session_state.ndsi_results
        st.session_state.step4_done = False
        st.warning("Feature selection reset. You can now try again.")

    if reduce_features:
        col1, col2 = st.columns([10, 1])
        with col1:
            reduction_method = st.radio("Choose reduction method", ["Feature Extraction", "Feature Selection"])
        with col2:
            tooltip("Feature Extraction creates new features from existing ones. Feature Selection picks the most important existing features")
        
        if reduction_method == "Feature Extraction":
            col1, col2 = st.columns([10, 1])
            with col1:
                extraction_method = st.selectbox("Choose extraction method", ["PCA", "Time Series", "t-SNE"])
            with col2:
                tooltip("PCA: Principal Component Analysis reduces dimensions while preserving variance. Time Series: Extracts features from temporal data. t-SNE: Non-linear technique for complex data")
            
            show_method_info(extraction_method, show_educational_info)
            
            if extraction_method == "PCA":
                col1, col2 = st.columns([10, 1])
                with col1:
                    variance_percentage = st.slider("Select the variance percentage to keep", 70.0, 100.0, 98.0, step=1.0)
                with col2:
                    tooltip("Higher values keep more information but fewer dimensions are reduced. A value of 95% means keeping 95% of the original information")
                
                if st.button("Apply PCA", key="apply_pca"):
                    try:
                        with st.spinner("Applying PCA..."):
                            reduced_data, total_cols_before, total_cols_after, cum_var = functions.perform_pca(
                                st.session_state.data, target_column, categorical_columns, variance_percentage
                            )
                        
                        st.success(f"PCA applied successfully! Reduced from {total_cols_before} to {total_cols_after} features.")
                        st.session_state.data = reduced_data
                        st.session_state.step4_done = True
                        
                        with st.expander("PCA Results"):
                            st.dataframe(reduced_data.head(), width=700, height=200)
                            functions.plot_cumulative_variance(cum_var, variance_percentage)
                            st.markdown(f"""
                            **Number of Features:**
                            - **Before PCA:** {total_cols_before}
                            - **After PCA:** {total_cols_after}
                            """)
                    except Exception as e:
                        st.error(f"Error applying PCA: {str(e)}")
                        st.error("Please ensure your data is properly preprocessed and contains numeric features.")
            
            elif extraction_method == "Time Series":
                if st.button("Apply Time Series Feature Extraction", key="apply_timeseries"):
                    try:
                        with st.spinner("Applying Time Series Feature Extraction..."):
                            st.session_state.data = functions.time_series_feature_extraction(
                                st.session_state.data, target_column, categorical_columns
                            )
                        
                        st.success("Time Series Feature Extraction applied successfully!")
                        st.session_state.step4_done = True
                        
                        with st.expander("Time Series-Based Feature Extraction Results"):
                            st.dataframe(st.session_state.data, width=700, height=200)
                    except Exception as e:
                        st.error(f"Error applying Time Series Feature Extraction: {str(e)}")
                        st.error("Please ensure your data contains appropriate time series features.")
            
            elif extraction_method == "t-SNE":
                col1, col2 = st.columns([10, 1])
                with col1:
                    n_components = st.slider("Select the number of t-SNE components", 2, 10, 2, step=1)
                with col2:
                    tooltip("Number of dimensions to reduce your data to. Usually 2 or 3 for visualization purposes")
                
                if st.button("Apply t-SNE", key="apply_tsne"):
                    try:
                        with st.spinner("Applying t-SNE (this may take a while)..."):
                            st.session_state.data = functions.perform_tsne(
                                st.session_state.data, target_column, categorical_columns, n_components=n_components
                            )
                        
                        st.success(f"t-SNE applied successfully! Reduced to {n_components} components.")
                        st.session_state.step4_done = True
                        
                        with st.expander("t-SNE Results"):
                            st.dataframe(st.session_state.data.head(), width=700, height=200)
                    except Exception as e:
                        st.error(f"Error applying t-SNE: {str(e)}")
                        st.error("Please ensure your data is properly preprocessed and contains numeric features.")
        
        elif reduction_method == "Feature Selection":
            col1, col2 = st.columns([10, 1])
            with col1:
                selection_method = st.selectbox("Choose selection method", ["NDSI", "SelectKBest", 'RFE'])
            with col2:
                tooltip("NDSI: Normalized Difference Spectral Index. SelectKBest: Selects top k features based on statistical tests. RFE: Recursive Feature Elimination")
            
            show_method_info(selection_method, show_educational_info)
            
            if selection_method == "NDSI":
                if st.button("Analyze NDSI", key="analyze_ndsi"):
                    try:
                        with st.spinner("Analyzing NDSI correlations..."):
                            df_results = functions.NDSI_pearson(st.session_state.data, categorical_columns, target_column)
                            df_results['p_value'] = df_results['p_value'].apply(lambda p: f"{p:.2e}" if p < 1e-3 else round(p, 3))
                        
                        st.session_state.ndsi_results = df_results
                        st.success("NDSI analysis completed!")
                        
                        st.subheader("NDSI Pearson Results")
                        st.dataframe(df_results)
                    except Exception as e:
                        st.error(f"Error analyzing NDSI: {str(e)}")
                        st.error("Please ensure your data contains appropriate spectral features.")
                
                if 'ndsi_results' in st.session_state:
                    col1, col2 = st.columns([10, 1])
                    with col1:
                        threshold = st.slider('Threshold', min_value=0.0, max_value=1.0, value=0.4)
                    with col2:
                        tooltip("Minimum correlation threshold. Higher values will select only strongly correlated features")
                    
                    col1, col2 = st.columns([10, 1])
                    with col1:
                        max_distance = st.slider('Max Distance', min_value=1, max_value=50, value=10)
                    with col2:
                        tooltip("Maximum band separation distance to consider")
                    
                    if st.button("Apply NDSI Selection", key="apply_ndsi"):
                        try:
                            with st.spinner("Applying NDSI feature selection..."):
                                top_bands_list = functions.display_ndsi_heatmap(st.session_state.ndsi_results, threshold, max_distance)
                                final_ndsi_df = functions.calculate_ndsi(st.session_state.data, top_bands_list)
                                
                                # Safely concatenate with categorical and target columns
                                concat_cols = []
                                if categorical_columns:
                                    for col in categorical_columns:
                                        if col in st.session_state.data.columns:
                                            concat_cols.append(st.session_state.data[col])
                                
                                if target_column in st.session_state.data.columns:
                                    concat_cols.append(st.session_state.data[target_column])
                                
                                if concat_cols:
                                    st.session_state.data = pd.concat([final_ndsi_df] + concat_cols, axis=1)
                                else:
                                    st.session_state.data = final_ndsi_df
                            
                            st.success(f"NDSI selection applied! Number of columns: {len(st.session_state.data.columns)}")
                            st.session_state.step4_done = True
                            
                            with st.expander("NDSI Results"):
                                st.info(f"Number of columns after NDSI calculation: {len(st.session_state.data.columns)}")
                                st.dataframe(st.session_state.data)
                        except Exception as e:
                            st.error(f"Error applying NDSI selection: {str(e)}")
            
            elif selection_method in ["SelectKBest", "RFE"]:
                col1, col2 = st.columns([10, 1])
                with col1:
                    # Use current feature count, with fallback
                    try:
                        current_features = [col for col in st.session_state.data.columns if col != target_column and col not in categorical_columns]
                        max_features = len(current_features) if current_features else 1
                        default_features = min(max_features, len(features) if 'features' in locals() else max_features)
                    except:
                        max_features = 1
                        default_features = 1
                    
                    num_features = st.slider("Select the number of top features", 1, max_features, default_features, step=1)
                with col2:
                    tooltip("Number of most important features to keep")
                
                button_text = f"Apply {selection_method}"
                if st.button(button_text, key=f"apply_{selection_method.lower()}"):
                    try:
                        with st.spinner(f"Applying {selection_method}..."):
                            if selection_method == "SelectKBest":
                                st.session_state.data = functions.perform_select_kbest(
                                    st.session_state.data, target_column, categorical_columns, k=num_features
                                )
                            else:  # RFE
                                st.session_state.data = functions.perform_rfe(
                                    st.session_state.data, target_column, categorical_columns, num_features
                                )
                        
                        st.success(f"{selection_method} applied successfully! Selected {num_features} features.")
                        st.session_state.step4_done = True
                        
                        with st.expander(f"{selection_method} Results"):
                            st.dataframe(st.session_state.data, width=700, height=200)
                    except Exception as e:
                        st.error(f"Error applying {selection_method}: {str(e)}")
                        st.error("Please ensure your data contains sufficient numeric features for selection.")
    
    # --- Show data after feature selection/extraction ---
    if st.session_state.get('step4_done', False):
        with st.expander("View Data After Feature Selection/Extraction"):
            st.dataframe(st.session_state.data)

    # Train-test split
    st.header("Step 6: Model Training")
    col1, col2 = st.columns([10, 1])
    with col1:
        split_percentage = st.slider("Select the train-test split percentage", 0.1, 0.9, 0.7)
    with col2:
        tooltip("Percentage of data to use for training. The rest will be used for testing")

    # Run Model button
    if st.button("Run Model") or st.session_state.get('models_evaluated', False):
        # If user skipped feature engineering, mark step4_done as True so progress advances
        if not st.session_state.get('step4_done', False):
            st.session_state.step4_done = True
        st.session_state.step5_done = True
        try:
            # Separate target column to prevent it from being dropped
            target = st.session_state.data[target_column]
            data = st.session_state.data.drop(columns=[target_column])

            # Data processing
            if 'missing_values_option' in locals():
                method_map = {
                    "Mean Imputation": "mean",
                    "Median Imputation": "median",
                    "Mode Imputation": "mode",
                    "KNN Imputation": "knn",
                    "Drop Rows": "drop_rows",
                    "Drop Columns": "drop_cols"
                }
                method = method_map[missing_values_option]
                k = k_neighbors if method == "knn" else 5
                data = functions.impute_missing_values(data, method=method, k=k)

            # Data Smoothing
            if 'smoothing_method' in locals():
                method_map = {
                    "Moving Average": "moving_average",
                    "Exponential Smoothing": "exponential",
                    "Savitzky-Golay": "savgol",
                    "LOWESS": "lowess"
                }
                method = method_map[smoothing_method]
                params = {}
                if method in ["moving_average", "savgol"]:
                    params["window_size"] = window_size
                elif method == "exponential":
                    params["alpha"] = alpha
                elif method == "lowess":
                    params["window_size"] = int(frac * len(data))
                data = functions.smooth_data(data, method=method, **params)

            # Log preprocessing choices
            preprocessing_choices = {
                "missing_values_method": missing_values_option if 'missing_values_option' in locals() else None,
                "smoothing_method": smoothing_method if 'smoothing_method' in locals() else None,
                "normalization_method": normalization_method if normalize_data else None,
                "categorical_encoding": categorical_encoding_method if encode_categorical_variables and has_categorical_columns else None
            }

            # Encoding
            if has_categorical_columns:
                numerical_columns = list(set(data.columns) - set(categorical_columns))
                numerical_columns = list(set(numerical_columns).intersection(data.select_dtypes(include=['number']).columns))
                
                if encode_categorical_variables and categorical_encoding_method and categorical_columns:
                    if categorical_encoding_method == "OneHotEncoder":
                        encoded_data = pd.get_dummies(data[categorical_columns], columns=categorical_columns, dtype=float)
                        data = pd.concat([data.drop(categorical_columns, axis=1), encoded_data], axis=1)
                    elif categorical_encoding_method == "LabelEncoder":
                        label_encoder = LabelEncoder()
                        for col in categorical_columns:
                            data[col] = label_encoder.fit_transform(data[col])

            # Normalization
            if normalize_data:
                numerical_columns = data.columns
                scaler = MinMaxScaler() if normalization_method == "MinMaxScaler" else StandardScaler()
                data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

            # Add the target column back to the processed data before displaying it
            if 'target_column' in st.session_state:
                data[st.session_state.target_column] = target
            
            st.subheader("Processed Data:")
            st.write(data)

            # Prepare data for modeling
            if 'X_train' not in st.session_state.state:
                # The target is already in the 'data' DataFrame, so we just need to sample and split
                data = data.sample(frac=1, random_state=42)
                X = data.drop(st.session_state.target_column, axis=1)
                y = data[st.session_state.target_column]
                X.columns = X.columns.astype(str)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-split_percentage, random_state=42)
                st.session_state.state['X_train'], st.session_state.state['X_test'], st.session_state.state['y_train'], st.session_state.state['y_test'] = X_train, X_test, y_train, y_test

            X_train, X_test, y_train, y_test = st.session_state.state['X_train'], st.session_state.state['X_test'], st.session_state.state['y_train'], st.session_state.state['y_test']

            if not st.session_state.models_evaluated:
                st.subheader("Regression Models Performance on Training Set")
                with st.spinner("Running multiple regression models on your data. This may take up to a few minutes depending on dataset size and model complexity..."):
                    models_df = functions.evaluate_regression_models(X_train, X_test, y_train, y_test)
                models_df = models_df.sort_values(by='R-Squared', ascending=False)
                st.dataframe(models_df[['R-Squared','RMSE','Time Taken']])
                
                # Show educational info about all models if enabled
                if show_educational_info:
                    with st.expander("📚 Learn about Machine Learning Models", expanded=False):
                        st.markdown("""
                        ### Popular Regression Models Explained:
                        
                        **Tree-Based Models:**
                        - **Random Forest**: Ensemble of decision trees that vote on predictions. Reduces overfitting and provides feature importance.
                        - **XGBoost**: Optimized gradient boosting that often achieves the best performance on structured data.
                        - **LightGBM**: Fast gradient boosting framework good for large datasets.
                        - **Gradient Boosting**: Sequentially builds trees to correct previous errors.
                        
                        **Linear Models:**
                        - **Linear Regression**: Simple straight-line fit, very interpretable.
                        - **Ridge/Lasso**: Regularized linear models that prevent overfitting.
                        - **Elastic Net**: Combines benefits of both Ridge and Lasso.
                        
                        **Support Vector Models:**
                        - **SVR**: Uses support vectors to find optimal prediction boundaries.
                        - **NuSVR**: SVR variant with more interpretable parameters.
                        
                        **Neural Networks:**
                        - **MLPRegressor**: Multi-layer neural network for complex patterns.
                        
                        **Other Models:**
                        - **K-Nearest Neighbors**: Predicts based on similar training examples.
                        - **AdaBoost**: Ensemble method that focuses on difficult samples.
                        - **Bagging**: Reduces variance by averaging multiple models.
                        
                        [Learn more about Machine Learning Models](https://en.wikipedia.org/wiki/Machine_learning)
                        """)
                
                st.session_state.models_df = models_df
                st.session_state.top_5_models = models_df.head(15).index.tolist()
                st.session_state.models_evaluated = True

            models_df = st.session_state.models_df
            top_5_models = st.session_state.top_5_models

            # Cross-validation folds control
            col1, col2 = st.columns([10, 1])
            with col1:
                num_cv_folds = st.number_input(
                    "Number of Cross-Validation Folds for Tuning",
                    min_value=3,
                    max_value=10,
                    value=5,
                    step=1,
                    help="Number of folds to use for cross-validation during hyperparameter tuning"
                )
            with col2:
                tooltip("Cross-validation splits data into k folds for robust hyperparameter tuning. More folds = more robust but slower training.")

            # Store CV folds in session state for future use and logging
            st.session_state.num_cv_folds = num_cv_folds

            best_model = st.selectbox("Select a model for hyperparameter tuning:", 
                                      top_5_models, 
                                      key='model_selection')

            # Show educational info for the selected model
            show_method_info(best_model, show_educational_info)

            if best_model != st.session_state.best_model:
                st.session_state.best_model = best_model
                
                if best_model in functions.model_functions_dict:
                    with st.spinner(f"Tuning hyperparameters for {best_model}..."):
                        tuning_function = functions.model_functions_dict[best_model]
                        best_estimator, best_params = tuning_function(X_train, y_train)
                    
                    if best_estimator is not None:
                        st.subheader(f"Hyperparameter Tuning Results for {best_model}")
                        st.write("Best Parameters:", best_params)
                        model_obj = best_estimator
                    else:
                        st.warning(f"Hyperparameter tuning failed for {best_model}. Using default model.")
                        model_obj = functions.get_model_object(best_model)
                        model_obj.fit(X_train, y_train)
                else:
                    st.info(f"No hyperparameter tuning function available for {best_model}. Using default model.")
                    model_obj = functions.get_model_object(best_model)
                    model_obj.fit(X_train, y_train)

                st.session_state.model_obj = model_obj

            model_obj = st.session_state.model_obj
            y_pred = model_obj.predict(X_test)
            
            # Model evaluation
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            
            # Log model performance
            iteration_data = {
                **preprocessing_choices,
                "model_name": best_model,
                "metrics": {
                    "mae": mae,
                    "mse": mse,
                    "rmse": rmse,
                    "r2": r2
                },
                "split_percentage": split_percentage,
                "cv_folds": st.session_state.get('num_cv_folds', 5)  # Default to 5 if not set
            }
            st.session_state.model_logger.log_iteration(iteration_data)
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("MAE", f"{mae:.4f}")
            col2.metric("MSE", f"{mse:.4f}")
            col3.metric("RMSE", f"{rmse:.4f}")
            col4.metric("R²", f"{r2:.4f}")

            # Add download button for logs
            if len(st.session_state.model_logger.logs) > 0:
                st.session_state.model_logger.save_logs()
                
                # Read the CSV file and create a download button
                with open('model_iterations.csv', 'r') as f:
                    st.download_button(
                        label="Download Model Iterations Log (CSV)",
                        data=f.read(),
                        file_name="model_iterations.csv",
                        mime="text/csv"
                    )
                
                # Display the log as a table in the app
                st.subheader("Model Iteration History")
                log_df = st.session_state.model_logger.get_logs_df()
                st.dataframe(log_df)

            # True vs Predicted Plot
            st.subheader("True Vs Prediction Plot")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode='markers', 
                                     marker=dict(color='blue', opacity=0.5),
                                     name='Predicted vs True Values'))
            fig.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()],
                                     mode='lines', line=dict(color='black', dash='dash'),
                                     name='1:1 Line'))
            fig.update_layout(title='True vs Predicted Values',
                              xaxis_title='True Values',
                              yaxis_title='Predicted Values',
                              showlegend=True,
                              width=800,
                              height=600)
            st.plotly_chart(fig)
            
            # Feature Importance
            st.subheader("Feature Importance")
            
            # Show educational info for feature importance
            if show_educational_info:
                with st.expander("📚 Understanding Feature Importance", expanded=False):
                    st.markdown("""
                    **Feature Importance** shows which variables most influence your model's predictions.
                    
                    - **Higher values** = More important features
                    - **Lower values** = Less important features
                    
                    This helps you understand what drives your predictions and can guide feature selection for future models.
                    
                    [Learn more about Feature Importance](https://en.wikipedia.org/wiki/Feature_importance)
                    """)
            
            importances = model_obj.feature_importances_ if hasattr(model_obj, 'feature_importances_') else permutation_importance(model_obj, X_train, y_train, n_repeats=10, random_state=42).importances_mean
            indices = np.argsort(importances)[::-1]
            names = [X_train.columns[i] for i in indices]
            importance_values = [importances[i] for i in indices]

            fig = go.Figure(go.Pie(labels=names, values=importance_values,
                                   textinfo='label+percent', hole=0.3,
                                   marker=dict(colors=plt.cm.Set3.colors, line=dict(color='white', width=2))))

            fig.update_layout(title_text="Feature Importance", margin=dict(l=0, r=0, t=60, b=0))
            st.plotly_chart(fig)
            
            # PDP Plots
            st.subheader("PDP Plots")
            
            # Show educational info for PDP plots
            if show_educational_info:
                with st.expander("📚 Understanding Partial Dependence Plots (PDP)", expanded=False):
                    st.markdown("""
                    **Partial Dependence Plots** show how each feature affects predictions while accounting for other features.
                    
                    - **X-axis**: Feature values
                    - **Y-axis**: Average prediction
                    - **Trend**: Shows if the feature has a positive, negative, or complex relationship with predictions
                    
                    [Learn more about Partial Dependence Plots](https://en.wikipedia.org/wiki/Partial_dependence_plot)
                    """)
            
            with st.spinner("Generating PDP plots..."):
                num_features = len(X_train.columns)
                max_plots_per_row = 5
                num_rows = (num_features + max_plots_per_row - 1) // max_plots_per_row
                num_cols = min(num_features, max_plots_per_row)
                    
                fig, axs = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 5 * num_rows), constrained_layout=True)
                axs = axs.flatten() if num_features > 1 else [axs]
                
                for j, selected_feature in enumerate(X_train.columns):
                    features_info = {
                        "features": [selected_feature],
                        "kind": "average",
                    }
                
                    display = PartialDependenceDisplay.from_estimator(
                        model_obj,
                        X_train,
                        **features_info,
                        ax=axs[j],
                        line_kw={"lw": 3}
                    )
                
                    axs[j].set_title(f"PDP for {selected_feature}")
                    axs[j].set_xlabel(selected_feature)
                    axs[j].set_ylabel(f"Partial Dependence for {target_column}")
                    axs[j].set_facecolor('#f0f0f0')
            
                fig.suptitle(f"Partial Dependence of {target_column} on Selected Features", y=1.02)
                plt.tight_layout()
                st.pyplot(fig)

            # SHAP Value Plot
            st.subheader("SHAP Value Plot")
            
            # Show educational info for SHAP values
            if show_educational_info:
                with st.expander("📚 Understanding SHAP Values", expanded=False):
                    st.markdown("""
                    **SHAP (SHapley Additive exPlanations)** values explain how each feature contributes to predictions.
                    
                    - **Red bars**: Features pushing predictions higher
                    - **Blue bars**: Features pushing predictions lower
                    - **Bar length**: Magnitude of feature's impact
                    - **Position**: Feature value (high/low)
                    
                    This provides both global and local interpretability of your model.
                    
                    [Learn more about SHAP Values](https://en.wikipedia.org/wiki/Shapley_value)
                    """)
            
            with st.spinner("Calculating SHAP values..."):
                explainer = shap.Explainer(model_obj, X_test)
                shap_values = explainer(X_test)
            
                fig, ax = plt.subplots(figsize=(10, 6))
                shap.plots.beeswarm(shap_values, order=shap_values.abs.max(0))
                st.pyplot(fig)

            # Save Model
            if st.button("Save Model"):
                model_filename = f"{best_model}_model.pkl"
                serialized_model = pickle.dumps(model_obj)
            
                st.download_button(
                    label="Download Model",
                    data=serialized_model,
                    file_name=model_filename,
                    mime="application/octet-stream",
                )
            
                st.success(f"Model saved (available for download).")

        except Exception as e:
            st.error(f"Error during model training and evaluation: {str(e)}")

else:
    st.info("Please upload a data file to continue.")
