import streamlit as st
from PIL import Image


st.set_page_config(page_title="Loan Analytics Portal", page_icon="💰", layout="wide")

st.markdown("""
    <style>
    .main-title {
        font-size: 42px;
        font-weight: 800;
        color: #2E86C1;
        text-align: center;
        margin-bottom: 0px;
    }
    .subtitle {
        text-align: center;
        font-size: 20px;
        color: #34495E;
        margin-top: 0px;
    }
    .instructions {
        background-color: #EAF2F8;
        border-radius: 12px;
        padding: 20px;
        margin-top: 25px;
        text-align: center;
        font-size: 18px;
    }
    </style>
""", unsafe_allow_html=True)

# Main title
st.markdown('<div class="main-title">💳 Loan Eligibility Visual Analytics</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Fast, consistent, and data-driven loan decisions</div>', unsafe_allow_html=True)

# Instructions box
st.markdown(
'<div class="instructions">📂 To get started, please attach your <b>Loan Dataset (.csv)</b> file using the sidebar on the <b>Analysis</b> page.</div>',
unsafe_allow_html=True
)


# Divider and navigation info
st.markdown("---")
st.sidebar.title("Navigation")
st.sidebar.success("Use the sidebar to switch between pages: 'Analysis' and 'Predictor'.")

col1, col2, col3 = st.columns([1,2,1])
with col2:
    st.markdown("""
    ### 🧭 Overview
    - **Analysis Page:** Explore historical loan data visually.
    - **Predictor Page:** Input applicant details to predict loan approval.


    ### 🧠 About the Project
    This app demonstrates the power of Visual Analytics tools.
    Built using **Python**, **scikit-learn**, and **Streamlit**, it combines interpretability and simplicity.
    The underlying model is a **Logistic Regression** trained on a loan dataset.
    """)


st.markdown("---")
st.caption("Developed by Cinta Carot Gracia, October 2025")