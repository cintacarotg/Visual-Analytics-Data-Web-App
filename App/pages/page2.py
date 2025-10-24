import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

MODEL_PATH = os.path.join("..", "Data", "loan_app_artifacts.pkl")
MODEL_INFO = (
    "Model: Logistic Regression trained on the loan dataset. "
    "Uses saved OneHotEncoder + StandardScaler for preprocessing."
)

# --- LOAD MODEL ---
def load_model(model_path=MODEL_PATH):
    if not os.path.exists(model_path):
        st.error(f"Model file not found at: {model_path}")
        return None
    with open(model_path, "rb") as f:
        model_artifact = pickle.load(f)
    return model_artifact

# --- PREPROCESS INPUT ---
def preprocess_input(df, model_artifact):
    df2 = df.copy()

    numeric_cols = model_artifact["numeric_cols"]
    categorical_cols = model_artifact["categorical_cols"]
    encoder = model_artifact["encoder"]
    scaler = model_artifact.get("scaler")
    model_columns = model_artifact["model_columns_ordered"]

    # Compute TotalIncome if present in numeric_cols
    if "TotalIncome" in numeric_cols:
        df2["TotalIncome"] = df2["ApplicantIncome"] + df2["CoapplicantIncome"]

    # Numeric preprocessing
    for col in numeric_cols:
        if col not in df2.columns:
            df2[col] = 0  # fill missing numeric columns with 0
        df2[col] = pd.to_numeric(df2[col], errors="coerce")
    if scaler is not None:
        df2[numeric_cols] = scaler.transform(df2[numeric_cols])

    # Categorical preprocessing
    df2[categorical_cols] = df2[categorical_cols].fillna("Missing")
    cat_encoded = encoder.transform(df2[categorical_cols])
    cat_encoded_df = pd.DataFrame(cat_encoded, columns=encoder.get_feature_names_out(), index=df2.index)

    # Combine numeric + categorical
    X = pd.concat([df2[numeric_cols], cat_encoded_df], axis=1)

    # Reorder to model's expected columns
    X = X.reindex(columns=model_columns, fill_value=0)
    return X.values

# --- STREAMLIT ---
st.set_page_config(page_title="Loan Approval Predictor", page_icon="🤖", layout="centered")

st.markdown("""
<style>
.title { font-size: 40px; font-weight: 700; color: #2E86C1; text-align: center; margin-bottom: 0px; }
.subtitle { text-align: center; font-size: 18px; color: #34495E; margin-bottom: 20px; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🤖 Loan Approval Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Enter applicant details below to estimate approval probability</div>', unsafe_allow_html=True)

# Load model artifact
model_artifact = load_model()
if model_artifact is None:
    st.stop()
model = model_artifact["model"]

# --- FORM ---
with st.form("applicant_form"):
    st.subheader("📋 Applicant Information")
    col1, col2 = st.columns(2)
    with col1:
        Gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        Married = st.selectbox("Married", ["Yes", "No"])
        Dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
        Education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        Self_Employed = st.selectbox("Self Employed", ["No", "Yes"])
        ApplicantIncome = st.number_input("Applicant Income (USD)", min_value=0.0, value=3000.0, step=100.0)
    with col2:
        CoapplicantIncome = st.number_input("Coapplicant Income (USD)", min_value=0.0, value=0.0, step=100.0)
        LoanAmount = st.number_input("Loan Amount (in thousands)", min_value=0.0, value=120.0, step=10.0)
        Loan_Amount_Term = st.selectbox("Loan Term (months)", [360, 120, 180, 240, 300, 60])
        Credit_History = st.selectbox("Credit History (1 = good, 0 = poor)", [1.0, 0.0])
        Property_Area = st.selectbox("Property Area", ["Urban", "Rural", "Semiurban"])
    submitted = st.form_submit_button("🔍 Predict Loan Approval")

if submitted:
    input_df = pd.DataFrame([{
        "Gender": Gender,
        "Married": Married,
        "Dependents": Dependents,
        "Education": Education,
        "Self_Employed": Self_Employed,
        "ApplicantIncome": ApplicantIncome,
        "CoapplicantIncome": CoapplicantIncome,
        "LoanAmount": LoanAmount,
        "Loan_Amount_Term": Loan_Amount_Term,
        "Credit_History": Credit_History,
        "Property_Area": Property_Area
    }])
    try:
        X_proc = preprocess_input(input_df, model_artifact)
        prob = model.predict_proba(X_proc)[0, 1]
        label = "✅ Approved" if prob >= 0.5 else "❌ Rejected"
        color = "green" if prob >= 0.5 else "red"
        st.markdown(f"<h3 style='text-align:center; color:{color};'>{label}</h3>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align:center; font-size:22px;'>Approval Probability: <b>{prob:.2%}</b></p>", unsafe_allow_html=True)
    except Exception as e:
        st.exception(e)
        st.error("Prediction failed — check model and input values.")

st.markdown("---")
st.caption(MODEL_INFO)
