# Hyperparameter Tuning Enhancement Implementation Plan

## Overview
This document outlines the implementation plan for adding:
1. **Metric Selection** for hyperparameter tuning
2. **Customizable Hyperparameter Ranges** for top models

## Current State Analysis

### Issue 1: Fixed Scoring Metric
- All `RandomizedSearchCV` calls use default scoring (R² for regression)
- No option for users to choose MAE, MSE, RMSE, etc.
- Line 701 example: `RandomizedSearchCV(lasso_cv, param_distributions=param_dist, n_iter=50, cv=5, random_state=42)`

### Issue 2: Fixed Hyperparameter Ranges
- All tuning functions have hardcoded parameter distributions
- Users cannot customize ranges based on their domain knowledge
- Example: `tune_rf_regressor` has fixed ranges for all hyperparameters (lines 1779-1792)

## Implementation Strategy

### Phase 1: Metric Selection (Universal)
**Location**: Before model selection in `ML_Framework_code.py` (~line 1498)

**UI Addition**:
```python
# Add before "Select a model for hyperparameter tuning"
col1, col2 = st.columns([10, 1])
with col1:
    tuning_metric = st.selectbox(
        "Select metric for hyperparameter tuning:",
        options=["R² (Default)", "MAE (Mean Absolute Error)", "MSE (Mean Squared Error)", "RMSE (Root Mean Squared Error)"],
        help="This metric will be used to evaluate model performance during hyperparameter search"
    )
with col2:
    tooltip("R²: Goodness of fit. MAE: Average absolute error. MSE: Squared error (penalizes large errors). RMSE: Square root of MSE")

# Convert to sklearn scoring
scoring_map = {
    "R² (Default)": "r2",
    "MAE (Mean Absolute Error)": "neg_mean_absolute_error",
    "MSE (Mean Squared Error)": "neg_mean_squared_error",
    "RMSE (Root Mean Squared Error)": "neg_root_mean_squared_error"
}
st.session_state.tuning_scoring = scoring_map[tuning_metric]
```

### Phase 2: Hyperparameter Customization (Top Models Only)
**Scope**: Focus on 4 most-used models to keep UI manageable
- RandomForestRegressor
- XGBRegressor
- LGBMRegressor
- GradientBoostingRegressor

**UI Pattern** (after model selection):
```python
if best_model in ["RandomForestRegressor", "XGBRegressor", "LGBMRegressor", "GradientBoostingRegressor"]:
    with st.expander("🔧 Advanced: Customize Hyperparameter Ranges (Optional)", expanded=False):
        st.info("Leave fields empty to use default ranges. Only modify if you understand the parameters.")
        
        if best_model == "RandomForestRegressor":
            col1, col2 = st.columns(2)
            with col1:
                n_estimators_min = st.number_input("n_estimators (min)", value=200, min_value=10)
                n_estimators_max = st.number_input("n_estimators (max)", value=2000, min_value=10)
                max_depth_options = st.multiselect("max_depth options", [10, 20, 30, 40, 50, None], default=[10, 20, 30, 40, 50])
            with col2:
                min_samples_split = st.multiselect("min_samples_split", [2, 5, 10, 15], default=[2, 5, 10])
                min_samples_leaf = st.multiselect("min_samples_leaf", [1, 2, 4, 8], default=[1, 2, 4])
            
            # Store in session_state
            st.session_state.custom_rf_params = {
                "n_estimators": [int(x) for x in np.linspace(n_estimators_min, n_estimators_max, 10)],
                "max_depth": max_depth_options,
                "min_samples_split": min_samples_split,
                "min_samples_leaf": min_samples_leaf
            }
```

### Phase 3: Modify Tuning Functions
**Approach**: Add optional parameters to tuning functions

**Example for `tune_rf_regressor`**:
```python
def tune_rf_regressor(X_train, y_train, custom_params=None, scoring='r2'):
    try:
        rf_regressor = RandomForestRegressor()
        
        # Use custom params if provided, otherwise use defaults
        if custom_params:
            param_dist = {**default_rf_params, **custom_params}
        else:
            param_dist = default_rf_params
        
        # Use custom scoring
        random_search = RandomizedSearchCV(
            rf_regressor, 
            param_distributions=param_dist, 
            n_iter=50, 
            cv=5, 
            scoring=scoring,  # NEW
            random_state=42
        )
        random_search.fit(X_train, y_train)
        
        return random_search.best_estimator_, random_search.best_params_
```

### Phase 4: Update Function Calls
**In `ML_Framework_code.py`** (~line 1510):
```python
# Pass scoring and custom params
tuning_function = functions.model_functions_dict[best_model]
custom_params = st.session_state.get(f'custom_{best_model}_params', None)
scoring = st.session_state.get('tuning_scoring', 'r2')

best_estimator, best_params = tuning_function(
    X_train, y_train, 
    custom_params=custom_params, 
    scoring=scoring
)
```

## Implementation Priority

### HIGH PRIORITY (Implement First):
1. ✅ Metric selection - applies to ALL models
2. ✅ Update top 4 model tuning functions to accept `scoring` parameter
3. ✅ Update function calls to pass scoring parameter

### MEDIUM PRIORITY (Implement Second):
4. Hyperparameter customization for RandomForestRegressor
5. Hyperparameter customization for XGBRegressor
6. Hyperparameter customization for LGBMRegressor
7. Hyperparameter customization for GradientBoostingRegressor

### LOW PRIORITY (Future Enhancement):
8. Add hyperparameter customization for additional models as needed
9. Add "preset" configurations (conservative, balanced, aggressive)
10. Save/load custom hyperparameter configurations

## Files to Modify

1. **functions.py**
   - Modify 4 key tuning functions to accept `custom_params` and `scoring`
   - Lines to modify: ~1773 (RF), ~1917 (XGB), ~1877 (LGBM), ~1693 (GradientBoosting)

2. **ML_Framework_code.py**
   - Add metric selection UI (~line 1485)
   - Add hyperparameter customization UI (~line 1502)
   - Update function call to pass parameters (~line 1510)

3. **ML_Framework_code.py** (Educational Info)
   - Add explanation of different metrics
   - Add tooltips for hyperparameters

## Testing Strategy

### Test Cases:
1. Default behavior (no custom params, R² metric) - should work as before
2. Change metric to MAE - verify RandomizedSearchCV uses MAE
3. Custom RF hyperparameters - verify custom ranges are used
4. Edge case: invalid ranges - should show error message
5. Multiple models - verify settings apply correctly to each

### Test Datasets:
- Small dataset (wine) - quick testing
- Medium dataset (UCI Credit Card) - realistic use case
- Large dataset - performance testing

## Benefits

### For Users:
✅ **Domain Knowledge Integration**: Adjust ranges based on dataset characteristics
✅ **Metric Flexibility**: Choose metric aligned with business goals
✅ **Experimental Control**: Test different configurations
✅ **Educational**: Learn about hyperparameters through interaction

### For Application:
✅ **Backward Compatible**: Defaults work without changes
✅ **Focused Scope**: Only top models to avoid UI complexity
✅ **Optional Feature**: Advanced users only, doesn't impact beginners
✅ **Clean Architecture**: Minimal changes to existing functions

## Risks & Mitigation

### Risk 1: UI Complexity
**Mitigation**: Use expandable sections, focus on top 4 models only

### Risk 2: Invalid Parameter Ranges
**Mitigation**: Add validation, clear error messages, sensible defaults

### Risk 3: Performance Impact
**Mitigation**: Custom ranges don't increase search iterations (still n_iter=50)

### Risk 4: User Confusion
**Mitigation**: Clear tooltips, default values visible, optional feature

## Next Steps

1. **Review this plan** - ensure approach is sound
2. **Implement Phase 1** (metric selection) - universal benefit, low risk
3. **Test Phase 1** thoroughly
4. **Implement Phase 2** (RF hyperparameters only) - validate approach
5. **Extend to other 3 models** if Phase 2 successful
6. **Document in README** with screenshots

## Estimated Implementation Time

- Phase 1 (Metric Selection): 1-2 hours
- Phase 2 (One Model Customization): 2-3 hours
- Phase 2 (All 4 Models): 6-8 hours total
- Testing & Documentation: 2-3 hours

**Total**: 10-13 hours for complete implementation

## Alternative: Simpler Approach

If full implementation is too complex, consider:

### Minimal Version:
1. ✅ Add metric selection (universal)
2. ✅ Add simple "Aggressive/Balanced/Conservative" preset selector
3. ❌ Skip individual hyperparameter customization

This provides 80% of value with 20% of complexity.

