import streamlit as st
import pandas as pd
from utils import load_model, simple_preprocess, MODEL_INFO

st.set_page_config(page_title="Loan Approval Predictor", page_icon="🤖", layout="centered")

# --- HEADER ---
st.markdown("""
    <style>
    .title {
        font-size: 40px;
        font-weight: 700;
        color: #2E86C1;
        text-align: center;
        margin-bottom: 0px;
    }
    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #34495E;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🤖 Loan Approval Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Enter applicant details below to estimate approval probability</div>', unsafe_allow_html=True)

# --- Load Model ---
model = load_model()  # loads from Data/loan_app_artifacts.pkl by default

# --- Applicant Input Form ---
with st.form(key='applicant_form'):
    st.subheader("📋 Applicant Information")

    col1, col2 = st.columns(2)
    with col1:
        Gender = st.selectbox('Gender', ['Male', 'Female', 'Other'])
        Married = st.selectbox('Married', ['Yes', 'No'])
        Dependents = st.selectbox('Dependents', ['0', '1', '2', '3+'])
        Education = st.selectbox('Education', ['Graduate', 'Not Graduate'])
        Self_Employed = st.selectbox('Self Employed', ['No', 'Yes'])
        ApplicantIncome = st.number_input('Applicant Income (USD)', min_value=0.0, value=3000.0, step=100.0)

    with col2:
        CoapplicantIncome = st.number_input('Coapplicant Income (USD)', min_value=0.0, value=0.0, step=100.0)
        LoanAmount = st.number_input('Loan Amount (in thousands)', min_value=0.0, value=120.0, step=10.0)
        Loan_Amount_Term = st.selectbox('Loan Term (months)', [360, 120, 180, 240, 300, 60])
        Credit_History = st.selectbox('Credit History (1 = good, 0 = poor)', [1.0, 0.0])
        Property_Area = st.selectbox('Property Area', ['Urban', 'Rural', 'Semiurban'])

    submitted = st.form_submit_button('🔍 Predict Loan Approval')

# --- Prediction Logic ---
if submitted:
    row = {
        'Gender': Gender,
        'Married': Married,
        'Dependents': Dependents,
        'Education': Education,
        'Self_Employed': Self_Employed,
        'ApplicantIncome': ApplicantIncome,
        'CoapplicantIncome': CoapplicantIncome,
        'LoanAmount': LoanAmount,
        'Loan_Amount_Term': Loan_Amount_Term,
        'Credit_History': Credit_History,
        'Property_Area': Property_Area
    }

    X = pd.DataFrame([row])

    if model is None:
        st.error("⚠️ Model file not found. Please ensure 'Data/loan_app_artifacts.pkl' exists.")
    else:
        try:
            # Always use internal simple_preprocess (no external preprocessor)
            X_proc = simple_preprocess(X)

            # Predict probability (ensure model supports predict_proba)
            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(X_proc)[0, 1]
            else:
                # fallback: use predict() (less informative)
                pred = model.predict(X_proc)[0]
                prob = float(pred)

            pred_label = "✅ Approved" if prob >= 0.5 else "❌ Rejected"
            color = "green" if prob >= 0.5 else "red"

            st.markdown(f"<h3 style='text-align:center; color:{color};'>{pred_label}</h3>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align:center; font-size:22px;'>Predicted Approval Probability: <b>{prob:.2%}</b></p>", unsafe_allow_html=True)

            st.info("Interpretation: A higher probability indicates a greater chance of loan approval.")

        except Exception as e:
            st.exception(e)
            st.error("Prediction failed — ensure your model was trained to accept the preprocessing output this app provides.")

st.markdown("---")
st.caption(MODEL_INFO)