# Explainable Machine Learning for Credit Risk Prediction

A comparative study of model accuracy and interpretability using SHAP, on the UCI German Credit dataset.

**Student:** Zaara Riyaz Khan | **Guide:** Dr. Atul Srivastava | Amity University Uttar Pradesh (B.Tech Minor Project)

**Live app:** https://syhrqqyniry7xyfchoappbo.streamlit.app

## Overview
This project compares four credit risk models (Logistic Regression, Decision Tree, Random Forest, XGBoost) on both predictive performance and interpretability. SHAP is used for global explanations (summary plots) and local explanations (waterfall plots for individual applicants). The best model and its explanations are deployed in a Streamlit app.

## Results (test set, 200 applicants)
| Model | Accuracy | Precision | Recall | F1-score | AUC-ROC |
|---|---|---|---|---|---|
| Logistic Regression | 0.760 | 0.575 | 0.767 | 0.657 | 0.810 |
| Decision Tree | 0.615 | 0.400 | 0.567 | 0.469 | 0.654 |
| Random Forest | 0.755 | 0.596 | 0.567 | 0.581 | 0.768 |
| XGBoost | 0.630 | 0.412 | 0.550 | 0.471 | 0.692 |

Logistic Regression, the most interpretable model, also performed best on F1, Recall and AUC-ROC, so no accuracy-vs-interpretability trade-off was observed on this (small) dataset.

## Method
- **Data:** UCI German Credit (1,000 records, 20 attributes, about 70:30 good/bad). Bad credit is the positive class.
- **Preprocessing:** 80/20 stratified split; winsorisation of credit amount and age; ordinal encoding for checking and savings account; one-hot encoding for other categorical features; StandardScaler for numeric features; SMOTE on training data only.
- **Tuning:** GridSearchCV (5-fold, F1 scoring) with SMOTE applied inside each fold.
- **Explainability:** SHAP LinearExplainer (Logistic Regression) and TreeExplainer (Decision Tree, Random Forest, XGBoost).

## Files
| File | Purpose |
|---|---|
| `credit_risk_shap_colab.ipynb` | Full pipeline: EDA, preprocessing, models, metrics, SHAP, export |
| `app.py` | Streamlit app: applicant form, risk prediction, SHAP waterfall |
| `credit_artifacts.pkl` | Preprocessor, trained models, feature names (exported from the notebook) |
| `requirements.txt` | Exact library versions used for training (needed to load the models) |

## Run locally
Use Python 3.12 (the pinned library versions do not install on Python 3.14).
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Push `app.py`, `credit_artifacts.pkl` and `requirements.txt` to a GitHub repo.
2. On share.streamlit.io choose *Create app*, select the repo, and set the main file to `app.py`.
3. In *Advanced settings* set the Python version to 3.12, then deploy.

## Disclaimer
Academic project, not for real lending decisions. The dataset contains sensitive attributes (age, sex/personal status, foreign worker) that would need a fairness review in any real system.
