"""
Optimized Caching Strategy for Streamlit ML Framework
Provides strategic caching for expensive operations while avoiding caching of dynamic data
"""

import streamlit as st
import pandas as pd
import numpy as np
import hashlib
from typing import Any, Dict, List, Optional, Callable, Union
from functools import wraps
import time


def create_data_hash(data: Union[pd.DataFrame, np.ndarray, List, str]) -> str:
    """Create a stable hash for data objects"""
    try:
        if isinstance(data, pd.DataFrame):
            return hashlib.md5(pd.util.hash_pandas_object(data).values).hexdigest()
        elif isinstance(data, np.ndarray):
            return hashlib.md5(data.tobytes()).hexdigest()
        elif isinstance(data, (list, str)):
            return hashlib.md5(str(data).encode()).hexdigest()
        else:
            return hashlib.md5(str(data).encode()).hexdigest()
    except Exception:
        return hashlib.md5(str(data).encode()).hexdigest()


def cache_expensive_computation(ttl: Optional[int] = None, show_spinner: bool = True):
    """
    Cache expensive computations that don't depend on frequently changing data.
    
    Use this for:
    - File loading operations
    - Model training 
    - Feature extraction that depends on static data
    - Large computations with stable inputs
    
    Do NOT use for:
    - DataFrame operations that change frequently
    - UI state dependent operations
    - Operations that modify session state
    """
    def decorator(func):
        @wraps(func)
        @st.cache_data(ttl=ttl, show_spinner=show_spinner)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator


def cache_model_objects():
    """
    Cache heavy, unchanging objects like pre-trained models or configuration objects.
    
    Use this for:
    - Pre-trained model instances
    - Large configuration objects
    - Expensive object creation that happens once per session
    """
    def decorator(func):
        @wraps(func)
        @st.cache_resource
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator


def conditional_cache(condition_func: Callable[[], bool]):
    """
    Conditionally cache based on a function that determines if caching should occur.
    
    Args:
        condition_func: Function that returns True if caching should be enabled
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if condition_func():
                # Use caching
                @st.cache_data
                def cached_func(*args, **kwargs):
                    return func(*args, **kwargs)
                return cached_func(*args, **kwargs)
            else:
                # Direct execution without caching
                return func(*args, **kwargs)
        return wrapper
    return decorator


class PerformanceMonitor:
    """Monitor and track performance of operations"""
    
    def __init__(self):
        if 'performance_logs' not in st.session_state:
            st.session_state.performance_logs = []
    
    def log_operation(self, operation: str, duration: float, cached: bool = False):
        """Log an operation's performance"""
        log_entry = {
            'operation': operation,
            'duration': duration,
            'cached': cached,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        st.session_state.performance_logs.append(log_entry)
        
        # Keep only last 100 entries
        if len(st.session_state.performance_logs) > 100:
            st.session_state.performance_logs.pop(0)
    
    def get_performance_summary(self) -> pd.DataFrame:
        """Get performance summary as DataFrame"""
        if not st.session_state.performance_logs:
            return pd.DataFrame()
        
        df = pd.DataFrame(st.session_state.performance_logs)
        summary = df.groupby('operation').agg({
            'duration': ['mean', 'min', 'max', 'count'],
            'cached': 'sum'
        }).round(3)
        
        summary.columns = ['avg_duration', 'min_duration', 'max_duration', 'call_count', 'cached_calls']
        summary['cache_hit_rate'] = (summary['cached_calls'] / summary['call_count'] * 100).round(1)
        
        return summary.reset_index()


def performance_monitor(operation_name: str):
    """Decorator to monitor performance of operations"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            monitor = PerformanceMonitor()
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                monitor.log_operation(operation_name, duration)
                return result
            except Exception as e:
                duration = time.time() - start_time
                monitor.log_operation(f"{operation_name}_error", duration)
                raise e
        return wrapper
    return decorator


# ============================================================================
# OPTIMIZED CACHING IMPLEMENTATIONS FOR SPECIFIC USE CASES
# ============================================================================

@cache_expensive_computation(ttl=3600)  # Cache for 1 hour
def load_file_data(file_content: bytes, filename: str) -> pd.DataFrame:
    """
    Load file data with caching based on file content hash.
    This is safe to cache because file content is immutable.
    """
    import io
    
    if filename.endswith('.csv'):
        return pd.read_csv(io.BytesIO(file_content))
    elif filename.endswith(('.xlsx', '.xls')):
        return pd.read_excel(io.BytesIO(file_content))
    else:
        raise ValueError(f"Unsupported file format: {filename}")


@cache_expensive_computation(ttl=1800)  # Cache for 30 minutes
def perform_pca_cached(data_hash: str, target_column: str, categorical_columns: List[str], 
                      variance_percentage: float) -> tuple:
    """
    Cached PCA computation. Only caches based on data hash to ensure consistency.
    """
    # This function should only be called with hashed data identifiers
    # The actual implementation will be in the main functions.py
    pass


@cache_expensive_computation(ttl=1800)
def calculate_ndsi_cached(data_hash: str, top_bands_list: List[tuple]) -> pd.DataFrame:
    """
    Cached NDSI calculation based on data hash and band combinations.
    """
    pass


@cache_model_objects()
def get_model_instances() -> Dict[str, Any]:
    """
    Get cached instances of ML models.
    These are expensive to create but don't change during the session.
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.svm import SVR
    
    return {
        'RandomForestRegressor': RandomForestRegressor(),
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(),
        'SVR': SVR()
    }


@performance_monitor("data_preprocessing")
def preprocess_data_monitored(data: pd.DataFrame, operations: List[str]) -> pd.DataFrame:
    """
    Example of monitored data preprocessing.
    This tracks performance but doesn't cache since data changes frequently.
    """
    result = data.copy()
    for operation in operations:
        # Simulate processing
        time.sleep(0.1)  # Remove in actual implementation
    return result


def show_cache_performance():
    """Display cache performance dashboard"""
    monitor = PerformanceMonitor()
    summary = monitor.get_performance_summary()
    
    if summary.empty:
        st.info("No performance data available yet.")
        return
    
    st.subheader("🚀 Performance Dashboard")
    
    # Overall stats
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_operations = summary['call_count'].sum()
        st.metric("Total Operations", total_operations)
    
    with col2:
        avg_cache_hit_rate = summary['cache_hit_rate'].mean()
        st.metric("Avg Cache Hit Rate", f"{avg_cache_hit_rate:.1f}%")
    
    with col3:
        avg_duration = summary['avg_duration'].mean()
        st.metric("Avg Operation Time", f"{avg_duration:.3f}s")
    
    # Detailed table
    st.subheader("Operation Details")
    st.dataframe(summary, use_container_width=True)
    
    # Performance tips
    with st.expander("💡 Performance Tips"):
        st.markdown("""
        **Green Indicators (Good Performance):**
        - High cache hit rates (>70%)
        - Low average durations (<1s for most operations)
        - Consistent performance across calls
        
        **Red Flags:**
        - Very low cache hit rates for expensive operations
        - Increasing average durations over time
        - High variance in operation times
        
        **Optimization Strategies:**
        1. Ensure expensive computations use appropriate caching
        2. Avoid caching operations on frequently changing data
        3. Monitor operations that consistently take >2 seconds
        """)


# ============================================================================
# CACHE INVALIDATION UTILITIES
# ============================================================================

def clear_specific_cache(function_name: str):
    """Clear cache for a specific function"""
    try:
        if hasattr(st.cache_data, 'clear'):
            st.cache_data.clear()
        st.success(f"Cache cleared for {function_name}")
    except Exception as e:
        st.error(f"Error clearing cache: {e}")


def clear_all_caches():
    """Clear all Streamlit caches"""
    try:
        st.cache_data.clear()
        st.cache_resource.clear()
        st.success("All caches cleared successfully!")
    except Exception as e:
        st.error(f"Error clearing caches: {e}")


def create_cache_management_ui():
    """Create UI for cache management"""
    st.subheader("🗄️ Cache Management")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Clear Data Cache"):
            st.cache_data.clear()
            st.success("Data cache cleared!")
    
    with col2:
        if st.button("Clear Resource Cache"):
            st.cache_resource.clear()
            st.success("Resource cache cleared!")
    
    with col3:
        if st.button("Clear All Caches"):
            clear_all_caches()
    
    # Cache statistics
    st.subheader("Cache Statistics")
    show_cache_performance()


# ============================================================================
# SMART CACHING DECISIONS
# ============================================================================

def should_cache_operation(operation_type: str, data_size: int, is_data_static: bool) -> bool:
    """
    Intelligent decision making for whether an operation should be cached.
    
    Args:
        operation_type: Type of operation (e.g., 'file_load', 'data_transform', 'model_train')
        data_size: Size of data being processed
        is_data_static: Whether the data changes frequently
    
    Returns:
        bool: Whether to cache this operation
    """
    # Always cache file loading
    if operation_type == 'file_load':
        return True
    
    # Cache model training for medium to large datasets
    if operation_type == 'model_train' and data_size > 1000:
        return True
    
    # Don't cache data transformations on dynamic data
    if operation_type == 'data_transform' and not is_data_static:
        return False
    
    # Cache expensive feature engineering on static data
    if operation_type == 'feature_engineering' and is_data_static and data_size > 500:
        return True
    
    # Default to no caching for safety
    return False
