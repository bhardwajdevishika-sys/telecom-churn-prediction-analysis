# Telecom Customer Churn Prediction & Analysis

A machine learning project that predicts whether a telecom customer is likely to 
churn, based on their account details and service usage. The project also 
generates an interactive dashboard summarizing the analysis and model results.

---

## Dataset

Uses the **Telco Customer Churn** dataset (7,043 customers, 21 features), 
covering customer demographics, subscribed services, account information 
(tenure, contract type, payment method), and billing details.

Dataset source: [Kaggle — Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

---

## What This Project Does

1. **Data Cleaning** — Converts the Churn column to numeric, standardizes 
   redundant service categories (e.g. "No internet service" → "No"), and 
   removes rows with missing billing values.

2. **Exploratory Data Analysis** — Examines how churn rate varies across 
   gender, contract type, payment method, internet service, tech support, 
   and customer tenure.

3. **Model Training & Comparison** — Trains five classifiers (Logistic 
   Regression, SVM, KNN, Decision Tree, Random Forest) on a 70/30 train-test 
   split.

4. **Evaluation** — Reports Accuracy, F1-score, Precision, and Recall for 
   each model. Accuracy alone can be misleading here since the dataset is 
   imbalanced (~26.5% churn vs 73.5% non-churn) — a model predicting "no 
   churn" for everyone would still score ~73.5% accuracy without learning 
   anything useful.

5. **Feature Importance** — Uses the Random Forest model to identify which 
   customer attributes most strongly influence churn.

6. **Dashboard** — Combines all EDA charts and model comparison results into 
   a single interactive HTML dashboard built with Plotly.

---

## Results

| Model | Accuracy | F1-score | Precision | Recall |
|---|---|---|---|---|
| Logistic Regression | 81.14% | 61.36% | 65.7% | 57.56% |
| Support Vector Machine | 80.66% | 60.23% | 64.78% | 56.28% |
| Random Forest | 79.38% | 55.84% | 63.07% | 50.09% |
| K-Nearest Neighbor | 76.82% | 53.91% | 55.86% | 52.09% |
| Decision Tree | 73.27% | 49.28% | 48.67% | 49.91% |

**Top churn-driving features** (from Random Forest importance): TotalCharges, 
MonthlyCharges, tenure, Contract type (two-year), and Internet Service 
(Fiber optic) were the strongest predictors of churn.
---

## Key Insights

- Customers on **month-to-month contracts** churn at a much higher rate than 
  those on one-year or two-year contracts.
- **Newer customers** (low tenure) are far more likely to churn than 
  long-tenured ones — churn rate drops steadily as tenure increases.
- Customers paying via **electronic check** show a noticeably higher churn 
  rate compared to other payment methods.
- TotalCharges, MonthlyCharges, and tenure emerged as the strongest numeric predictors, more influential than most categorical service features."

---

## Project Structure

├── Customer_Churn_Prediction.py # Main script
├── Tel_Customer_Churn_Dataset.csv # Dataset
├── churn_dashboard.html # Generated dashboard (output)
├── requirements.txt
└── README.md

## How to Run

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
pip install -r requirements.txt
python Customer_Churn_Prediction.py
```

The script trains all models, prints evaluation metrics to the terminal, and 
opens the interactive dashboard in your default browser.

---

## Notes on This Implementation

While working on this project, I:
- Fixed a pandas compatibility issue where assigning integers into a 
  string-typed column raised a `TypeError` on newer pandas versions — 
  resolved using `.map()` instead of `.loc[]` assignment
- Added F1-score, Precision, and Recall metrics alongside accuracy to better 
  evaluate performance on this imbalanced dataset
- Added Random Forest feature importance analysis to surface actionable 
  churn drivers

---

## Acknowledgment

Built while learning from a publicly available churn prediction template, 
with the above analysis and fixes added on top.