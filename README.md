# Fraudulent Loan Application Detection

This project aims to build a machine learning model to detect fraudulent loan applications using historical data. The project includes data preprocessing, feature engineering, model building, and interpretability analysis. This repository also provides a detailed report documenting the methodology, findings, and potential improvements.

## Table of Contents
- [Problem Definition](#problem-definition)
- [Data Understanding](#data-understanding)
- [Approach](#approach)
  - [Data Preprocessing](#data-preprocessing)
  - [Feature Engineering](#feature-engineering)
  - [Model Building](#model-building)
  - [Model Interpretability](#model-interpretability)
  - [Model Performance and Error Analysis](#model-performance-and-error-analysis)
- [Results](#results)
- [Future Improvements](#future-improvements)
- [Setup Instructions](#setup-instructions)
- [File Structure](#file-structure)
- [License](#license)

## Problem Definition
The goal of this project is to predict whether a loan application is likely to be fraudulent based on the patterns in historical loan application data. Early detection of fraud can help financial institutions reduce risk and ensure secure lending practices.

## Data Understanding
We use a publicly available dataset representing loan application data. The dataset includes both numerical and categorical fields relevant to loan applications.

## Approach

### Data Preprocessing
1. **Handle Missing Values**: Missing values are handled by imputing with mean/median for numerical features and mode for categorical features.
2. **Outlier Removal**: Outliers in numerical features are removed using the IQR (Interquartile Range) method.
3. **Imbalanced Classes**: SMOTE (Synthetic Minority Over-sampling Technique) and SMOTENC (for categorical fields) are used to balance the classes.

### Feature Engineering
- New features are created based on domain knowledge to improve predictive performance, such as `Loan_to_Asset_Ratio` and `Income_per_Dependent`.
- These features are evaluated for their contribution to model performance.

### Model Building
- We use `RandomForest`, `XGBoost`, and `LightGBM` for binary classification.
- The models are evaluated using metrics like precision, recall, F1-score, and a confusion matrix.
- Cross-validation and hyperparameter tuning are performed to optimize model performance.
- Class imbalance is handled by setting appropriate class weights and using SMOTE.

### Model Interpretability
- **SHAP** values are used to interpret the XGBoost model and understand feature contributions to predictions.
- Feature importance is analyzed to identify key drivers of fraud predictions.

### Model Performance and Error Analysis
- Detailed error analysis is conducted to identify common misclassification patterns.
- Suggestions for improvement are provided, including additional features or data enrichment strategies.

## Results
The models achieved high accuracy and balanced F1-scores. The best-performing model was Random Forest, showing high interpretability and generalizability across different data subsets.

## Future Improvements
- Explore additional features that capture loan behavior over time.
- Consider ensemble techniques or deep learning models for further accuracy improvements.
- Add more sophisticated methods for handling rare categories in categorical features.

## Setup Instructions

### Prerequisites
- Python 3.7+
- Required libraries listed in `requirements.txt`.

### Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/NavyaMohanKK/DataScientistCodeAssessment.git
    cd DataScientistCodeAssessment
    ```

2. Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the code to train the model and generate results:
    ```bash
    python main.py
    ```

### Model Deployment
The trained models are saved as `.joblib` files for easy loading and use in deployment pipelines.
