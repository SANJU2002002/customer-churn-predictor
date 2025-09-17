
import streamlit as st
import joblib, pandas as pd, numpy as np
st.set_page_config(layout='wide', page_title='Churn Predictor Demo')
st.title("Churn Predictor — Interactive Demo")

@st.cache(allow_output_mutation=True)
def load_artifacts():
    xgb = joblib.load('models/xgb_pipeline_v1.pkl')
    sample = pd.read_csv('data/telco_clean.csv').sample(200, random_state=42)
    return xgb, sample

try:
    model, sample = load_artifacts()
except:
    st.error("Models/data not found. Place models in /models and data in /data")
    st.stop()

customer = st.selectbox("Pick a sample row index", sample.index.tolist())
row = sample.loc[[customer]]
st.write("Customer profile:")
st.dataframe(row.T)

if st.button("Predict churn"):
    proba = model.predict_proba(row)[:,1][0]
    st.metric("Churn probability", f"{proba:.2%}")
    if proba > 0.6:
        st.warning("High churn risk — recommend retention action")
    else:
        st.success("Low churn risk")
