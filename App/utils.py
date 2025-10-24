import os
import pickle
import pandas as pd
import numpy as np

# Base path (project root)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, 'Data', 'loan_app_artifacts.pkl')
DATA_PATH = os.path.join(BASE_DIR, 'Data', 'loan_dataset.csv')

MODEL_INFO = (
    "Model: Logistic Regression trained in the attached notebook. "
    "Model artifact loaded from '../Data/loan_app_artifacts.pkl'. "
    "This app uses an internal simple_preprocess() routine to transform user input."
)

def load_dataset(path=DATA_PATH):
    """Load dataset from ../Data/loan_dataset.csv"""
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception as e:
            print("Failed to read dataset:", e)
    return None

def load_model(model_path=MODEL_PATH):
    """Load trained model from ../Data/loan_app_artifacts.pkl"""
    model = None
    if os.path.exists(model_path):
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
        except Exception as e:
            print("⚠️ Failed loading model:", e)
    else:
        print(f"Model file not found at: {model_path}")
    return model

def simple_preprocess(df):
    """Simple preprocessing fallback"""
    df2 = df.copy()

    numeric_cols = [
        'ApplicantIncome', 'CoapplicantIncome',
        'LoanAmount', 'Loan_Amount_Term', 'Credit_History'
    ]
    for c in numeric_cols:
        if c in df2.columns:
            df2[c] = pd.to_numeric(df2[c], errors='coerce')

    df2['LoanAmount'] = df2['LoanAmount'].fillna(df2['LoanAmount'].median())
    df2['Loan_Amount_Term'] = df2['Loan_Amount_Term'].fillna(360)
    df2['Credit_History'] = df2['Credit_History'].fillna(1.0)

    cat_cols = [
        'Gender', 'Married', 'Dependents', 'Education',
        'Self_Employed', 'Property_Area'
    ]
    for c in cat_cols:
        if c in df2.columns:
            df2[c] = df2[c].fillna('Missing')

    if 'Dependents' in df2.columns:
        df2['Dependents'] = df2['Dependents'].replace('3+', 3)
        df2['Dependents'] = pd.to_numeric(df2['Dependents'], errors='coerce')

    df_cats = pd.get_dummies(df2[cat_cols], drop_first=True)
    X = pd.concat([df2[numeric_cols], df_cats], axis=1)

    return X.values