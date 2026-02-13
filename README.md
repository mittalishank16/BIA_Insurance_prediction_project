# Insurance Policy Response Prediction

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Machine Learning](https://img.shields.io/badge/ML-Classification-green)
![Status](https://img.shields.io/badge/Status-Completed-success)

## Project Overview
This project aims to optimize the insurance sales process by predicting customer interest in vehicle insurance policies. By leveraging machine learning, we identify high-probability leads, allowing for more efficient resource allocation and personalized marketing strategies.

The project addresses a significant **class imbalance** (only ~12% positive responses) using advanced resampling techniques and ensemble modeling.

---

## Key Insights from EDA
The Exploratory Data Analysis phase uncovered critical behavioral patterns:
* **Feature Significance**: Using **Cramér’s V**, we identified that `Vehicle_Damage` and `Previously_Insured` are the strongest indicators of customer interest.
* **Demographics**: Customers aged 30–50 with older vehicles showed a significantly higher response rate compared to younger drivers.
* **Data Quality**: The dataset was clean (no missing values) but highly skewed, particularly in the `Annual_Premium` feature, which required logarithmic transformation.

---

## Data Pipeline & Engineering
1. **Cleaning**: Dropped irrelevant identifiers (`id`) and handled duplicates.
2. **Feature Engineering**: 
   - **Encoding**: Binary encoding for `Gender`; Frequency encoding for `Region_Code` and `Policy_Sales_Channel`.
   - **Scaling**: Robust scaling for numerical variables to mitigate the impact of outliers.
3. **Resampling (SMOTETomek)**: Applied a hybrid technique combining **SMOTE** (Synthetic Minority Over-sampling Technique) and **Tomek Links** to clean the decision boundary between classes.

---

## Modeling & Optimization
### Why Tree-Based Models?
Initial tests with **Linear Models (Logistic Regression)** yielded poor results because:
- The data relationships are non-linear and high-dimensional.
- Linear models were overly sensitive to outliers in premiums.
- They failed to capture complex feature interactions (e.g., Age vs. Vehicle Damage).

**Final Selection:** **AdaBoost Classifier** was chosen for its superior ability to focus on hard-to-classify minority samples through sequential learning.

### Model Tracking
- **MLflow**: All experiments, hyperparameters, and metrics were logged using MLflow.
- **Optimization**: Hyperparameters were tuned using `RandomizedSearchCV`.

---

## Final Performance
The model was evaluated with a focus on **Recall** to ensure no potential leads are missed.

| Metric | Score |
| :--- | :--- |
| **Accuracy** | 79.29% |
| **Recall (Interested Class)** | **70.15%** |
| **F1-Macro Score** | 0.6604 |

### Classification Report (Test Set):
```text
              precision    recall  f1-score   support
           0       0.95      0.81      0.87     54252
           1       0.33      0.70      0.45      7390