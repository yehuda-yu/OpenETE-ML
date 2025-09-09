"""
Advanced State Management System for Streamlit ML Framework
Provides centralized DataFrame management with undo/redo functionality
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from copy import deepcopy
import hashlib
import json


@dataclass
class DataFrameSnapshot:
    """Represents a snapshot of the DataFrame at a specific point in time"""
    data: pd.DataFrame
    operation: str
    timestamp: str
    description: str
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert snapshot to dictionary for serialization"""
        return {
            'operation': self.operation,
            'timestamp': self.timestamp,
            'description': self.description,
            'metadata': self.metadata,
            'data_hash': self.get_data_hash()
        }
    
    def get_data_hash(self) -> str:
        """Generate a hash of the DataFrame for comparison"""
        return hashlib.md5(pd.util.hash_pandas_object(self.data).values).hexdigest()


class DataFrameStateManager:
    """
    Centralized state manager for DataFrame operations with undo/redo functionality.
    
    This class manages the DataFrame lifecycle, tracks changes, and provides
    robust undo/redo capabilities while ensuring optimal performance.
    """
    
    def __init__(self, max_history_size: int = 50):
        self.max_history_size = max_history_size
        self._initialize_state()
    
    def _initialize_state(self):
        """Initialize session state variables for DataFrame management"""
        # Core DataFrame storage
        if 'df_current' not in st.session_state:
            st.session_state.df_current = None
        
        # History management
        if 'df_history' not in st.session_state:
            st.session_state.df_history = []
        
        if 'df_current_index' not in st.session_state:
            st.session_state.df_current_index = -1
        
        # Operation tracking
        if 'df_operation_log' not in st.session_state:
            st.session_state.df_operation_log = []
        
        # File tracking
        if 'current_file_info' not in st.session_state:
            st.session_state.current_file_info = {}
        
        # Metadata
        if 'df_metadata' not in st.session_state:
            st.session_state.df_metadata = {
                'original_shape': None,
                'current_shape': None,
                'features': [],
                'target_column': None,
                'categorical_columns': [],
                'processing_steps': []
            }
    
    @property
    def current_dataframe(self) -> Optional[pd.DataFrame]:
        """Get the current active DataFrame"""
        return st.session_state.df_current
    
    @property
    def can_undo(self) -> bool:
        """Check if undo operation is possible"""
        return st.session_state.df_current_index > 0
    
    @property
    def can_redo(self) -> bool:
        """Check if redo operation is possible"""
        return st.session_state.df_current_index < len(st.session_state.df_history) - 1
    
    @property
    def history_size(self) -> int:
        """Get current history size"""
        return len(st.session_state.df_history)
    
    def load_initial_data(self, data: pd.DataFrame, filename: str) -> None:
        """
        Load initial data and reset the entire state.
        
        Args:
            data: The DataFrame to load
            filename: Name of the source file
        """
        # Clear existing state
        st.session_state.df_history.clear()
        st.session_state.df_operation_log.clear()
        st.session_state.df_current_index = -1
        
        # Set current data
        st.session_state.df_current = data.copy()
        
        # Update metadata
        st.session_state.df_metadata.update({
            'original_shape': data.shape,
            'current_shape': data.shape,
            'features': [],
            'target_column': None,
            'categorical_columns': [],
            'processing_steps': []
        })
        
        # Update file info
        st.session_state.current_file_info = {
            'filename': filename,
            'loaded_at': pd.Timestamp.now().isoformat(),
            'original_size': data.shape
        }
        
        # Create initial snapshot
        self._create_snapshot(
            operation='load_data',
            description=f'Loaded data from {filename}',
            metadata={'filename': filename, 'shape': data.shape}
        )
    
    def update_dataframe(self, 
                        new_data: pd.DataFrame, 
                        operation: str, 
                        description: str,
                        metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Update the DataFrame and create a new snapshot.
        
        Args:
            new_data: The new DataFrame
            operation: The operation name
            description: Human-readable description
            metadata: Additional operation metadata
        
        Returns:
            bool: True if update was successful
        """
        try:
            # Validate the new data
            if new_data is None or new_data.empty:
                st.error("Cannot update with empty DataFrame")
                return False
            
            # Update current DataFrame
            st.session_state.df_current = new_data.copy()
            
            # Update metadata
            st.session_state.df_metadata['current_shape'] = new_data.shape
            st.session_state.df_metadata['processing_steps'].append({
                'operation': operation,
                'description': description,
                'timestamp': pd.Timestamp.now().isoformat(),
                'shape_before': st.session_state.df_metadata.get('current_shape'),
                'shape_after': new_data.shape
            })
            
            # Create snapshot
            self._create_snapshot(operation, description, metadata or {})
            
            return True
            
        except Exception as e:
            st.error(f"Error updating DataFrame: {str(e)}")
            return False
    
    def _create_snapshot(self, operation: str, description: str, metadata: Dict[str, Any]):
        """Create a new snapshot and manage history"""
        # If we're not at the latest position, truncate future history
        if st.session_state.df_current_index < len(st.session_state.df_history) - 1:
            st.session_state.df_history = st.session_state.df_history[:st.session_state.df_current_index + 1]
        
        # Create new snapshot
        snapshot = DataFrameSnapshot(
            data=st.session_state.df_current.copy(),
            operation=operation,
            timestamp=pd.Timestamp.now().isoformat(),
            description=description,
            metadata=metadata
        )
        
        # Add to history
        st.session_state.df_history.append(snapshot)
        st.session_state.df_current_index = len(st.session_state.df_history) - 1
        
        # Maintain history size limit
        if len(st.session_state.df_history) > self.max_history_size:
            st.session_state.df_history.pop(0)
            st.session_state.df_current_index -= 1
        
        # Log the operation
        st.session_state.df_operation_log.append({
            'operation': operation,
            'description': description,
            'timestamp': pd.Timestamp.now().isoformat(),
            'snapshot_index': st.session_state.df_current_index
        })
    
    def undo(self) -> bool:
        """
        Undo the last operation.
        
        Returns:
            bool: True if undo was successful
        """
        if not self.can_undo:
            return False
        
        try:
            # Move to previous snapshot
            st.session_state.df_current_index -= 1
            snapshot = st.session_state.df_history[st.session_state.df_current_index]
            
            # Restore DataFrame
            st.session_state.df_current = snapshot.data.copy()
            
            # Update metadata
            st.session_state.df_metadata['current_shape'] = snapshot.data.shape
            
            return True
            
        except Exception as e:
            st.error(f"Error during undo: {str(e)}")
            return False
    
    def redo(self) -> bool:
        """
        Redo the next operation.
        
        Returns:
            bool: True if redo was successful
        """
        if not self.can_redo:
            return False
        
        try:
            # Move to next snapshot
            st.session_state.df_current_index += 1
            snapshot = st.session_state.df_history[st.session_state.df_current_index]
            
            # Restore DataFrame
            st.session_state.df_current = snapshot.data.copy()
            
            # Update metadata
            st.session_state.df_metadata['current_shape'] = snapshot.data.shape
            
            return True
            
        except Exception as e:
            st.error(f"Error during redo: {str(e)}")
            return False
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get current state status information"""
        current_snapshot = None
        if (st.session_state.df_current_index >= 0 and 
            st.session_state.df_current_index < len(st.session_state.df_history)):
            current_snapshot = st.session_state.df_history[st.session_state.df_current_index]
        
        return {
            'has_data': st.session_state.df_current is not None,
            'current_shape': st.session_state.df_metadata.get('current_shape'),
            'original_shape': st.session_state.df_metadata.get('original_shape'),
            'can_undo': self.can_undo,
            'can_redo': self.can_redo,
            'history_position': f"{st.session_state.df_current_index + 1}/{len(st.session_state.df_history)}",
            'current_operation': current_snapshot.operation if current_snapshot else None,
            'processing_steps_count': len(st.session_state.df_metadata.get('processing_steps', [])),
            'last_modified': current_snapshot.timestamp if current_snapshot else None
        }
    
    def get_operation_history(self) -> List[Dict[str, Any]]:
        """Get the complete operation history"""
        return [snapshot.to_dict() for snapshot in st.session_state.df_history]
    
    def update_feature_metadata(self, features: List[str], target: str, categorical: List[str]):
        """Update feature-related metadata"""
        st.session_state.df_metadata.update({
            'features': features,
            'target_column': target,
            'categorical_columns': categorical
        })
    
    def get_features(self) -> List[str]:
        """Get current feature list"""
        return st.session_state.df_metadata.get('features', [])
    
    def get_target_column(self) -> Optional[str]:
        """Get current target column"""
        return st.session_state.df_metadata.get('target_column')
    
    def get_categorical_columns(self) -> List[str]:
        """Get current categorical columns"""
        return st.session_state.df_metadata.get('categorical_columns', [])


def create_status_dashboard(state_manager: DataFrameStateManager):
    """Create a status dashboard showing current state information"""
    status = state_manager.get_current_status()
    
    if not status['has_data']:
        st.info("📁 No data loaded. Please upload a file to begin.")
        return
    
    # Main status row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "📊 Current Shape", 
            f"{status['current_shape'][0]} × {status['current_shape'][1]}" if status['current_shape'] else "N/A",
            delta=f"vs {status['original_shape'][0]} × {status['original_shape'][1]}" if status['original_shape'] else None
        )
    
    with col2:
        st.metric(
            "📝 Processing Steps", 
            status['processing_steps_count']
        )
    
    with col3:
        st.metric(
            "📚 History Position", 
            status['history_position']
        )
    
    with col4:
        if status['current_operation']:
            st.metric(
                "⚡ Current Operation", 
                status['current_operation']
            )
    
    # Undo/Redo controls
    col1, col2, col3 = st.columns([1, 1, 4])
    
    with col1:
        if st.button("↶ Undo", disabled=not status['can_undo'], key="undo_btn"):
            if state_manager.undo():
                st.success("✅ Undone successfully!")
                st.rerun()
            else:
                st.error("❌ Undo failed!")
    
    with col2:
        if st.button("↷ Redo", disabled=not status['can_redo'], key="redo_btn"):
            if state_manager.redo():
                st.success("✅ Redone successfully!")
                st.rerun()
            else:
                st.error("❌ Redo failed!")
    
    with col3:
        if status['last_modified']:
            st.caption(f"Last modified: {pd.to_datetime(status['last_modified']).strftime('%H:%M:%S')}")


# Global instance for the application
@st.cache_resource
def get_state_manager() -> DataFrameStateManager:
    """Get a singleton instance of the DataFrameStateManager"""
    return DataFrameStateManager()
