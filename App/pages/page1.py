import streamlit as st 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import load_dataset

st.set_page_config(page_title="Analysis", layout="wide")
st.title("Analysis — Exploratory Data Analysis")

# Set a pastel color palette
PASTEL_COLORS = ['#AEC6CF', '#FFB347', '#77DD77', '#F49AC2', '#FFD1DC', '#CFCFC4', '#B39EB5']

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

    # --- Categorical counts ---
    with col1:
        st.subheader("Categorical counts")
        cat_cols = df.select_dtypes(include=[object]).columns.tolist()
        if 'Loan_Status' in df.columns:
            cat_cols = [c for c in cat_cols if c != 'Loan_ID']
        for c in cat_cols:
            st.write(f"**{c}**")
            counts = df[c].value_counts(dropna=False)
            fig, ax = plt.subplots()
            counts.plot(kind='bar', color=PASTEL_COLORS[:len(counts)], edgecolor='k', alpha=0.8, ax=ax)
            ax.set_ylabel("Count")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
            st.pyplot(fig)

    # --- Numerical distributions ---
    with col2:
        st.subheader("Numerical distributions")
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        for c in num_cols:
            st.write(f"**{c}**")
            fig, ax = plt.subplots()
            df[c].dropna().plot(kind='hist', bins=30, color=PASTEL_COLORS[0], edgecolor='k', alpha=0.7, ax=ax)
            ax.set_xlabel(c)
            ax.set_ylabel("Frequency")
            st.pyplot(fig)

    # --- Bivariate exploration ---
    st.header("Bivariate exploration: Factors vs Loan_Status")
    if 'Loan_Status' in df.columns:
        st.subheader("Loan approval distribution")
        fig, ax = plt.subplots()
        counts = df['Loan_Status'].value_counts()
        counts.plot(kind='bar', color=PASTEL_COLORS[:len(counts)], edgecolor='k', alpha=0.8, ax=ax)
        ax.set_ylabel("Count")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        st.pyplot(fig)

        # Credit History vs Loan_Status
        st.subheader("Credit History vs Approval Rate")
        if 'Credit_History' in df.columns:
            pivot = df.pivot_table(index='Credit_History', columns='Loan_Status', aggfunc='size', fill_value=0)
            st.write(pivot)
            fig, ax = plt.subplots()
            pivot.plot(kind='bar', stacked=True, color=PASTEL_COLORS[:len(pivot.columns)], edgecolor='k', alpha=0.8, ax=ax)
            ax.set_ylabel("Count")
            ax.set_xlabel("Credit History")
            st.pyplot(fig)

        # Applicant Income boxplot
        if 'ApplicantIncome' in df.columns:
            st.subheader("Applicant Income by Loan_Status (boxplot)")
            fig, ax = plt.subplots()
            df.boxplot(column='ApplicantIncome', by='Loan_Status', ax=ax, patch_artist=True,
                       boxprops=dict(facecolor=PASTEL_COLORS[0], color='k'),
                       medianprops=dict(color='red'))
            ax.set_title("")
            ax.set_xlabel("Loan Status")
            ax.set_ylabel("Applicant Income")
            st.pyplot(fig)
        else:
            st.info("'Loan_Status' column not found — cannot show approval-based EDA.")

    # --- Correlations ---
    st.header("Quick correlations (numerical)")
    try:
        corr = df.select_dtypes(include=[np.number]).corr()
        fig, ax = plt.subplots()
        cax = ax.matshow(corr, cmap='Pastel1')
        fig.colorbar(cax)
        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=90)
        ax.set_yticklabels(corr.columns)
        st.pyplot(fig)
        st.dataframe(corr)
    except Exception:
        st.warning("Could not compute correlations — check numeric columns.")