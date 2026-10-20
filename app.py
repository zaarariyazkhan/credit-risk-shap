import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

st.set_page_config(page_title="Explainable Credit Risk", page_icon="💳", layout="wide")
FIELD_NAMES = {
    "status": "Checking account status",
    "duration": "Loan duration (months)",
    "credit_history": "Credit history",
    "purpose": "Purpose of loan",
    "credit_amount": "Credit amount (DM)",
    "savings": "Savings account / bonds",
    "employment": "Present employment since",
    "installment_rate": "Installment rate (% of disposable income)",
    "personal_status": "Personal status and sex",
    "other_debtors": "Other debtors / guarantors",
    "residence_since": "Present residence since (years)",
    "property": "Property",
    "age": "Age (years)",
    "other_installment_plans": "Other installment plans",
    "housing": "Housing",
    "existing_credits": "Existing credits at this bank",
    "job": "Job",
    "num_dependents": "People liable to provide maintenance",
    "telephone": "Telephone",
    "foreign_worker": "Foreign worker",
}

LABELS = {
    "status": {"A11": "< 0 DM", "A12": "0 to < 200 DM", "A13": ">= 200 DM / salary assigned",
               "A14": "No checking account"},
    "credit_history": {"A30": "No credits taken / all paid duly", "A31": "All credits at this bank paid duly",
                       "A32": "Existing credits paid duly till now", "A33": "Delay in paying in the past",
                       "A34": "Critical account / other credits elsewhere"},
    "purpose": {"A40": "Car (new)", "A41": "Car (used)", "A42": "Furniture / equipment",
                "A43": "Radio / television", "A44": "Domestic appliances", "A45": "Repairs",
                "A46": "Education", "A47": "Vacation", "A48": "Retraining", "A49": "Business",
                "A410": "Others"},
    "savings": {"A61": "< 100 DM", "A62": "100 to < 500 DM", "A63": "500 to < 1000 DM",
                "A64": ">= 1000 DM", "A65": "Unknown / no savings"},
    "employment": {"A71": "Unemployed", "A72": "< 1 year", "A73": "1 to < 4 years",
                   "A74": "4 to < 7 years", "A75": ">= 7 years"},
    "personal_status": {"A91": "Male: divorced/separated", "A92": "Female: divorced/separated/married",
                        "A93": "Male: single", "A94": "Male: married/widowed", "A95": "Female: single"},
    "other_debtors": {"A101": "None", "A102": "Co-applicant", "A103": "Guarantor"},
    "property": {"A121": "Real estate", "A122": "Building society savings / life insurance",
                 "A123": "Car or other", "A124": "Unknown / no property"},
    "other_installment_plans": {"A141": "Bank", "A142": "Stores", "A143": "None"},
    "housing": {"A151": "Rent", "A152": "Own", "A153": "For free"},
    "job": {"A171": "Unemployed / unskilled non-resident", "A172": "Unskilled resident",
            "A173": "Skilled employee / official", "A174": "Management / self-employed / highly qualified"},
    "telephone": {"A191": "None", "A192": "Yes"},
    "foreign_worker": {"A201": "Yes", "A202": "No"},
}


NUM_CONFIG = {
    "duration": (4, 72, 24, 1),
    "credit_amount": (250, 20000, 3000, 50),
    "installment_rate": (1, 4, 3, 1),
    "residence_since": (1, 4, 2, 1),
    "age": (18, 80, 35, 1),
    "existing_credits": (1, 4, 1, 1),
    "num_dependents": (1, 2, 1, 1),
}

MODEL_ORDER = ["Logistic Regression", "Decision Tree", "Random Forest", "XGBoost"]


@st.cache_resource
def load_artifacts():
    return joblib.load("credit_artifacts.pkl")


@st.cache_resource
def get_explainer(model_name: str):
    art = load_artifacts()
    model = art["models"][model_name]
    if model_name == "Logistic Regression":
        return shap.LinearExplainer(model, art["background"])
    return shap.TreeExplainer(model)


def pretty(feature: str) -> str:
    """Turn 'purpose_A40' into 'Purpose of loan: Car (new)'."""
    if feature in FIELD_NAMES:
        return FIELD_NAMES[feature]
    for col, mapping in LABELS.items():
        if feature.startswith(col + "_"):
            code = feature[len(col) + 1:]
            if code in mapping:
                return f"{FIELD_NAMES[col]}: {mapping[code]}"[:60]
    return feature


def explain_row(explainer, x_row: np.ndarray, names):
    """SHAP explanation for the 'bad credit' class, for one row."""
    sv = explainer.shap_values(x_row)
    if isinstance(sv, list):
        sv = sv[1]
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[:, :, 1]
    ev = np.atleast_1d(explainer.expected_value)
    base = ev[1] if len(ev) > 1 else ev[0]
    return shap.Explanation(values=sv[0], base_values=float(base), data=x_row[0], feature_names=names)



art = load_artifacts()
COLS = art["columns"]

st.title("💳 Explainable Credit Risk Prediction")
st.caption("Minor Project - Explainable ML for Credit Risk using SHAP (German Credit dataset). "
           "For academic demonstration only, not for real lending decisions.")

with st.sidebar:
    st.header("Settings")
    model_name = st.selectbox("Model", MODEL_ORDER, index=0,
                              help="Logistic Regression is the recommended model (best F1 / Recall / AUC-ROC).")
    threshold = st.slider("Risk threshold (probability of bad credit)", 0.1, 0.9, 0.5, 0.05)
    st.markdown("---")
    st.markdown("**Test-set results**")
    if "results" in art:
        st.dataframe(art["results"].round(3), use_container_width=True)

st.subheader("Applicant details")
with st.form("applicant_form"):
    left, right = st.columns(2)
    values = {}
    for i, col in enumerate(COLS):
        target = left if i % 2 == 0 else right
        with target:
            if col in NUM_CONFIG:
                lo, hi, default, step = NUM_CONFIG[col]
                values[col] = st.number_input(FIELD_NAMES[col], min_value=lo, max_value=hi,
                                              value=default, step=step)
            else:
                codes = list(LABELS[col].keys())
                values[col] = st.selectbox(FIELD_NAMES[col], codes,
                                           format_func=lambda c, col=col: LABELS[col][c])
    submitted = st.form_submit_button("Predict credit risk", type="primary")

if submitted:
    row = pd.DataFrame([values], columns=COLS)
    # same winsorisation as in training
    for c, (lo, hi) in art["caps"].items():
        row[c] = row[c].clip(lo, hi)

    x = art["preprocessor"].transform(row)
    model = art["models"][model_name]
    p_bad = float(model.predict_proba(x)[0, 1])
    is_risky = p_bad >= threshold

    st.subheader("Prediction")
    c1, c2 = st.columns([1, 2])
    c1.metric("Probability of bad credit", f"{p_bad:.1%}")
    if is_risky:
        c2.error(f"**High risk** - {model_name} predicts this applicant is likely a bad credit risk.")
    else:
        c2.success(f"**Low risk** - {model_name} predicts this applicant is likely a good credit risk.")

    st.markdown("**All models side by side**")
    comp = pd.DataFrame({
        "Model": MODEL_ORDER,
        "P(bad credit)": [float(art["models"][m].predict_proba(x)[0, 1]) for m in MODEL_ORDER],
    })
    comp["Prediction"] = np.where(comp["P(bad credit)"] >= threshold, "High risk", "Low risk")
    comp["P(bad credit)"] = comp["P(bad credit)"].map(lambda v: f"{v:.1%}")
    st.dataframe(comp, hide_index=True, use_container_width=True)

    st.subheader(f"Why? SHAP explanation ({model_name})")
    names = [pretty(f) for f in art["feature_names"]]
    exp = explain_row(get_explainer(model_name), x, names)
    plt.figure()
    shap.plots.waterfall(exp, max_display=12, show=False)
    st.pyplot(plt.gcf(), clear_figure=True)
    st.caption("Red bars push the prediction toward bad credit (higher risk); blue bars push toward good credit. "
               "Numeric feature values shown in the plot are standardised.")

    order = np.argsort(-np.abs(exp.values))[:5]
    st.markdown("**Top drivers for this applicant**")
    for i in order:
        direction = "increases" if exp.values[i] > 0 else "decreases"
        st.write(f"- {names[i]} {direction} risk ({exp.values[i]:+.2f})")
