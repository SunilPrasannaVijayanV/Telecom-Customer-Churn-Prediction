<div align="center">

# 📊 Telecom Customer Churn Prediction

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-150458.svg?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Status](https://img.shields.io/badge/Status-Completed-success.svg)](#-model-benchmarks--evaluation)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end Machine Learning project to analyze, visualize, and accurately predict customer churn for telecommunication service providers.

---

[Overview](#-overview) • [Key Objectives](#-key-objectives) • [Dataset Architecture](#-dataset-architecture) • [EDA Insights](#-eda-insights) • [Pipeline & Architecture](#-pipeline--architecture) • [Model Benchmarks](#-model-benchmarks--evaluation) • [Quick Start](#-quick-start) • [Author](#-author)

</div>

---

## 📌 Overview

**Customer Churn** occurs when customers or subscribers discontinue business with a service provider. In the telecommunications industry—where customers can easily switch between competing providers—annual churn rates range between **15% to 25%**.

Acquiring new customers can cost up to **5x to 7x more** than retaining existing ones. By leveraging predictive machine learning models, telecom companies can identify high-risk customers proactively and offer targeted retention incentives, preserving revenue and market share.

---

## 🎯 Key Objectives

1. **Quantify Churn Baseline**: Determine the overall percentage of churned vs. retained customers.
2. **Exploratory Data Analysis (EDA)**: Identify key operational, demographic, and financial drivers behind customer attrition.
3. **Automated ML Pipeline**: Implement an end-to-end Python pipeline (`run_pipeline.py`) handling cleaning, missing value imputation, encoding, scaling, training, cross-validation, and performance plotting.
4. **Model Benchmarking**: Train and evaluate 8 distinct machine learning algorithms (including ensemble models like Voting Classifier and AdaBoost) to determine the top performer.

---

## 📂 Dataset Architecture

The analysis utilizes the standard **Telco Customer Churn** dataset containing **7,043 customer records** and **21 attributes**:

| Feature Category | Features Included |
| :--- | :--- |
| **Demographics** | `gender`, `SeniorCitizen`, `Partner`, `Dependents` |
| **Services Signed Up** | `PhoneService`, `MultipleLines`, `InternetService` (DSL, Fiber optic, No), `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` |
| **Account Information** | `tenure` (months), `Contract` (Month-to-month, One year, Two year), `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges` |
| **Target Variable** | `Churn` (Yes = 1, No = 0) |

---

## 🔍 Key Exploratory Data Analysis (EDA) Insights

> Full EDA visualizations are saved in the [`output/`](./output/) directory.

* **Churn Baseline Rate**: **26.6%** of customers churned within the observed timeframe.
* **Contract Influence**: 
  * **Month-to-Month** contracts exhibit an alarming **~75%** churn share.
  * Customers with **One-Year** (~13%) or **Two-Year** (~3%) contracts are significantly more loyal.
* **Internet Service Dissatisfaction**: Customers on **Fiber Optic** plans experience a substantially higher churn rate compared to **DSL** subscribers, pointing to potential service dissatisfaction or price sensitivity.
* **Payment Methods**: Customers paying via **Electronic Check** churn at the highest rate, whereas automated bank transfers and credit card payments show strong retention.
* **Support Services**: Customers **without Tech Support** or **Online Security** are twice as likely to churn.
* **Tenure & Charges**: New customers (low tenure) and customers with higher monthly charges show the highest propensity to leave.

---

## ⚙️ Pipeline & Architecture

The workflow is encapsulated in a reproducible script [`run_pipeline.py`](./run_pipeline.py):

```
                        ┌────────────────────────┐
                        │   data.csv (7,043 x 21) │
                        └───────────┬────────────┘
                                    │
                        ┌───────────▼────────────┐
                        │ Data Preprocessing &   │
                        │ Missing Value Handling │
                        └───────────┬────────────┘
                                    │
                        ┌───────────▼────────────┐
                        │   One-Hot Encoding &   │
                        │    Standard Scaling    │
                        └───────────┬────────────┘
                                    │
                        ┌───────────▼────────────┐
                        │ 80/20 Train/Test Split │
                        │  5-Fold Stratified CV  │
                        └───────────┬────────────┘
                                    │
                        ┌───────────▼────────────┐
                        │ 8 Classification Models│
                        │ (LR, RF, AdaBoost, etc)│
                        └───────────┬────────────┘
                                    │
                        ┌───────────▼────────────┐
                        │ Automated Evaluation & │
                        │ Visualizations Plotting│
                        └────────────────────────┘
```

---

## 📈 Model Benchmarks & Evaluation

All models were evaluated using **5-Fold Stratified Cross-Validation** on the training set and validated against an independent **20% holdout test set (1,409 samples)**.

### Performance Summary Table

| Rank | Model | 5-Fold CV Acc (Mean ± Std) | Test Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 | **Logistic Regression** | 80.28% ± 1.22% | **80.55%** | 0.657 | 0.559 | **0.604** | 0.842 |
| 🥈 | **AdaBoost Classifier** | 80.72% ± 1.30% | **80.55%** | **0.670** | 0.527 | 0.590 | 0.843 |
| 🥉 | **Voting Classifier (Soft Ensemble)** | **81.01% ± 1.18%** | **80.13%** | 0.655 | 0.532 | 0.587 | **0.845** |
| 4 | **Gradient Boosting** | 80.25% ± 0.92% | 79.91% | 0.654 | 0.516 | 0.577 | 0.843 |
| 5 | **Random Forest** | 78.84% ± 1.35% | 78.78% | 0.626 | 0.497 | 0.554 | 0.825 |
| 6 | **K-Nearest Neighbors (KNN)** | 76.85% ± 0.78% | 76.65% | 0.560 | 0.564 | 0.562 | 0.793 |
| 7 | **Decision Tree** | 72.91% ± 1.76% | 74.24% | 0.515 | 0.495 | 0.505 | 0.663 |
| 8 | **Gaussian Naive Bayes** | 66.60% ± 1.14% | 65.58% | 0.427 | **0.866** | 0.572 | 0.809 |

---

### Best Model Classification Metrics (Logistic Regression)

```
              precision    recall  f1-score   support

    No Churn       0.85      0.89      0.87      1035
       Churn       0.66      0.56      0.60       374

    accuracy                           0.81      1409
   macro avg       0.75      0.73      0.74      1409
weighted avg       0.80      0.81      0.80      1409
```

---

## 🚀 Quick Start

### 1. Prerequisites & Dependencies

Ensure you have **Python 3.9+** installed. Install required dependencies:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### 2. Clone & Run Pipeline

Clone the repository and run the end-to-end pipeline:

```bash
git clone https://github.com/Pradnya1208/Telecom-Customer-Churn-prediction.git
cd Telecom-Customer-Churn-prediction

# Execute complete pipeline
python run_pipeline.py
```

Execution will train all models, generate benchmark reports in terminal, and output high-resolution charts to `output/`:
- `output/pipeline_confusion_matrix.png`
- `output/pipeline_model_comparison.png`

---

## 📁 Repository Structure

```
Telecom-Customer-Churn-prediction/
│── data.csv                       # Primary dataset (7,043 rows)
│── run_pipeline.py                # Standalone end-to-end ML script
│── README.md                      # Comprehensive project documentation
│── Scripts/
│   └── Customer churn prediction.ipynb  # Interactive Jupyter Notebook
│── output/                        # Visualizations & plot exports
│   ├── pipeline_confusion_matrix.png
│   ├── pipeline_model_comparison.png
│   └── ...
└── icons/                         # SVG icons for documentation
```

---

## 💡 Future Roadmap & Optimizations

- [ ] **Hyperparameter Tuning**: Apply `GridSearchCV` / `Optuna` fine-tuning on Gradient Boosting & XGBoost.
- [ ] **Handling Class Imbalance**: Implement **SMOTE** (Synthetic Minority Over-sampling Technique) to increase churn class recall.
- [ ] **Interactive Web Application**: Deploy model interface using **Streamlit** or **Flask API** for real-time predictions.
- [ ] **Customer Lifetime Value (CLV)**: Integrate financial impact forecasting per churn risk segment.

---

## 👤 Author & Acknowledgments

**Pradnya Patil**  
*Data Science & Machine Learning Practitioner*

- 🐙 **GitHub**: [@Pradnya1208](https://github.com/Pradnya1208)
- 💼 **LinkedIn**: [Pradnya Patil](https://www.linkedin.com/in/pradnya-patil-b049161ba/)
- 📊 **Tableau**: [Pradnya's Dashboards](https://public.tableau.com/app/profile/pradnya.patil3254#!/)
- 🐦 **Twitter**: [@Pradnya1208](https://twitter.com/Pradnya1208)

---

<div align="center">
⭐ <i>If you found this project helpful, please give it a star on GitHub!</i> ⭐
</div>
#   T e l e c o m - C u s t o m e r - C h u r n - P r e d i c t i o n  
 