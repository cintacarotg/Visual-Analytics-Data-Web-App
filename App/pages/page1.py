import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import load_dataset


st.set_page_config(page_title="Analysis", layout="wide")

st.title("Analysis — Exploratory Data Analysis")

# Dataset loader: upload or load default
st.sidebar.header("Load data")
uploaded = st.sidebar.file_uploader("Upload loan dataset CSV", type=["csv"])

if uploaded is not None:
    df = pd.read_csv(uploaded)
else:
    df = load_dataset()

if df is None:
    st.error("No dataset found. Upload a CSV using the sidebar, or place your dataset at 'Data/loan_dataset.csv'.")
else:
    st.markdown(f"**Dataset shape:** {df.shape}")
    if st.checkbox("Show raw data"):
        st.dataframe(df)

    st.header("Univariate exploration")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Categorical counts")
        cat_cols = df.select_dtypes(include=[object]).columns.tolist()
        if 'Loan_Status' in df.columns:
            cat_cols = [c for c in cat_cols if c!='Loan_ID']
        for c in cat_cols:
            st.write(f"**{c}**")
            counts = df[c].value_counts(dropna=False)
            st.bar_chart(counts)

    with col2:
        st.subheader("Numerical distributions")
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        for c in num_cols:
            st.write(f"**{c}**")
            fig, ax = plt.subplots()
            df[c].dropna().hist(bins=30)
            ax.set_xlabel(c)
            st.pyplot(fig)

    st.header("Bivariate exploration: Factors vs Loan_Status")
    if 'Loan_Status' in df.columns:
        grouped = df.groupby('Loan_Status')
        st.write("Loan approval distribution:")
        st.bar_chart(df['Loan_Status'].value_counts())

    # Example relationships
    st.subheader("Credit History vs Approval Rate")
    if 'Credit_History' in df.columns:
        pivot = df.pivot_table(index='Credit_History', columns='Loan_Status', aggfunc='size', fill_value=0)
        st.write(pivot)
        st.bar_chart(pivot)

    if 'ApplicantIncome' in df.columns:
        st.subheader("Applicant Income by Loan_Status (boxplot)")
        fig, ax = plt.subplots()
        df.boxplot(column='ApplicantIncome', by='Loan_Status', ax=ax)
        ax.set_title('')
        st.pyplot(fig)
    else:
        st.info("'Loan_Status' column not found in dataset — can't show approval-based EDA.")

    st.header('Quick correlations (numerical)')
    try:
        corr = df.select_dtypes(include=[np.number]).corr()
        st.dataframe(corr)
    except Exception:
        st.warning('Could not compute correlations — check numeric columns.')