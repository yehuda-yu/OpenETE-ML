# *OpenETE-ML*: An Open-Source End-to-End Machine Learning Web Application for Non-Programmers

## Overview

**OpenETE-ML** is a modern, open-source web application that enables users—especially non-programmers—to build, evaluate, and interpret machine learning regression models from start to finish. Designed for accessibility and scientific rigor, OpenETE-ML guides users through the entire ML workflow, from data upload and preprocessing to model selection, explainability, and export, all within an intuitive browser interface.

**Access the app online:**

👉 **[Launch OpenETE-ML in your browser](https://openete-ml.streamlit.app/)**

No installation or downloads required—just open the link and start building models!

This application is presented in the following scientific paper:

> **Yungstein, Y., & Helman, D. (2025). *OpenETE-ML*: An Open-Source End-To-End Machine Learning Web Application for Non-Programmers. [Submitted].**

If you use OpenETE-ML in your research, please cite this work (see citation below).

---

## Key Features

- **No Programming Required:** All steps are performed via a user-friendly web interface.
- **Cloud-Based:** Use the app instantly online—no setup needed.
- **Step-by-Step Workflow:** See below for a detailed breakdown of each step.
- **Educational Mode:** Toggle explanations and Wikipedia links for all major ML concepts and methods, making the app ideal for teaching and self-learning.
- **Progress Tracking:** Visual workflow progress bar and step-by-step sidebar.
- **Reproducibility:** All preprocessing and modeling choices are logged and exportable.

---

## Detailed Workflow Steps

OpenETE-ML guides you through the following steps, each with rich options and educational support:

### 1. Data Upload
- **What it does:** Upload your dataset in CSV or Excel format directly in the browser.
- **Options:**
  - Remove rows from the beginning (e.g., metadata or extra headers)
  - Instantly preview your data and basic statistics
- **Educational Mode:** Explains file formats, data structure, and best practices for data preparation.

### 2. Exploration & Cleaning
- **What it does:** Explore your dataset, identify missing values, and clean your data.
- **Options:**
  - View NaN counts per column
  - Replace specific values (e.g., -999, 0) or ranges with NaN
  - Remove or impute missing values (mean, median, mode, KNN, drop rows/columns)
- **Educational Mode:** Provides explanations for each cleaning method and when to use them.

### 3. Preprocessing
- **What it does:** Prepare your data for modeling.
- **Options:**
  - Normalize or standardize data (MinMaxScaler, StandardScaler)
  - Apply data smoothing (Moving Average, Exponential, Savitzky-Golay, LOWESS)
  - Encode categorical variables (OneHot, LabelEncoder)
  - Resample time series data by date column (various frequencies and aggregation methods)
- **Educational Mode:** Explains the importance of each preprocessing step and links to further reading.

### 4. Feature Engineering
- **What it does:** Reduce dimensionality or extract new features to improve model performance.
- **Options:**
  - Feature extraction: PCA (with variance slider), t-SNE (for visualization), time series feature extraction
  - Feature selection: NDSI (with heatmap and thresholding), SelectKBest, RFE
  - Instantly preview the transformed dataset
- **Educational Mode:** Describes each technique, its benefits, and how to interpret results.

### 5. Model Training
- **What it does:** Train and compare multiple regression models automatically.
- **Options:**
  - Choose train-test split percentage
  - Run a suite of models (Random Forest, XGBoost, SVR, and more)
  - Hyperparameter tuning for top models
- **Educational Mode:** Explains each model type, their strengths, and how to interpret performance metrics.

### 6. Evaluation & Explainability
- **What it does:** Analyze model performance and understand predictions.
- **Options:**
  - View metrics: MAE, MSE, RMSE, R²
  - Feature importance (interactive pie chart)
  - Partial Dependence Plots (PDP) for feature effects
  - SHAP value plots for model interpretability
  - Download experiment logs for reproducibility
- **Educational Mode:** Offers in-depth explanations of each metric and interpretability tool.

### 7. Export
- **What it does:** Download your trained model and logs for future use or publication.
- **Options:**
  - Download trained model as a file
  - Download CSV log of all modeling iterations and preprocessing choices
- **Educational Mode:** Explains the importance of reproducibility and how to use exported files.

---

## Scientific Citation

If you use OpenETE-ML in your research, please cite:

> Yungstein, Y., & Helman, D. (2025). *OpenETE-ML*: An Open-Source End-To-End Machine Learning Web Application for Non-Programmers. [Submitted].

**BibTeX:**
```bibtex
@article{yungstein2025openeteml,
  author    = {Yehuda Yungstein and David Helman},
  title     = {OpenETE-ML: An Open-Source End-To-End Machine Learning Web Application for Non-Programmers},
  year      = {2025},
  note      = {Submitted},
}
```

## License

This project is licensed under the MIT License.

---

## Authors
- Yehuda Yungstein
- David Helman

---

## Acknowledgments
We thank all contributors and the open-source community for their support.
