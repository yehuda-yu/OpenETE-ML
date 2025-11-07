# How to Revert Boruta Feature Selection

## Current State
**Branch deployed**: `main` (commit `d0bfae5`)
**Changes**: Added Boruta all-relevant feature selection method

## If You Want to Remove Boruta - Revert Instructions

### Option 1: Quick Revert (Recommended)
```bash
# Switch to main
git checkout main

# Revert to the commit before Boruta
git revert d0bfae5

# Push the revert
git push origin main
```

### Option 2: Hard Reset (Use with caution)
```bash
# Switch to main
git checkout main

# Reset to commit before Boruta implementation
git reset --hard 7c57803

# Force push (WARNING: This rewrites history)
git push --force origin main
```

### Option 3: Manual Removal
If you only want to remove Boruta but keep other changes:

1. **Remove from requirements.txt:**
   - Delete line: `boruta>=0.3`

2. **Remove from functions.py:**
   - Delete the `perform_boruta()` function (lines ~559-603)

3. **Remove from ML_Framework_code.py:**
   - Remove `'Boruta'` from selectbox options
   - Remove Boruta educational info
   - Remove the `elif selection_method == "Boruta":` block

Then commit and push:
```bash
git add requirements.txt functions.py ML_Framework_code.py
git commit -m "Remove Boruta feature selection"
git push origin main
```

## Commit History (Most Recent First)
- **d0bfae5**: Add Boruta all-relevant feature selection method ⬅️ **CURRENT**
- **7c57803**: Improve Excel file format handling and error messages
- **03d0f67**: Add revert instructions for cache removal changes
- **5ffbe77**: Remove @optimized_cache from data transformation functions
- **ee4e620**: Fix RFE function return type to prevent DataFrame conversion error

## What Was Added (commit d0bfae5)

### Files Modified:
1. **requirements.txt** - Added `boruta>=0.3` dependency
2. **functions.py** - Added `perform_boruta()` function (66 lines)
3. **ML_Framework_code.py** - Added UI components and Boruta handling logic

### Features Added:
- Boruta option in Feature Selection dropdown
- Configuration sliders for max_iter and alpha
- Educational information about all-relevant selection
- Error handling and user guidance

### Why You Might Want to Revert:
- ❌ Boruta is too slow for your datasets
- ❌ You prefer simpler feature selection methods
- ❌ Dependencies issues with boruta package
- ❌ You only need minimal optimal feature sets (RFE/SelectKBest sufficient)

### Why You Should Keep It:
- ✅ Provides comprehensive feature selection (all-relevant vs minimal)
- ✅ Better for understanding which factors contribute to predictions
- ✅ Complements existing RFE/SelectKBest methods
- ✅ Well-documented and tested implementation
- ✅ Minimal impact on existing functionality

## Testing Before Reverting
Try Boruta with your datasets first:
1. Use smaller datasets to test speed
2. Compare results with RFE/SelectKBest
3. Check if all-relevant selection helps your analysis
4. Monitor memory usage with large datasets

## Questions?
- Check GitHub issues
- Review commit `d0bfae5` for full changes
- See https://github.com/scikit-learn-contrib/boruta_py for Boruta documentation

