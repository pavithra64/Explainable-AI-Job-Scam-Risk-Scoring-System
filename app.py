import streamlit as st
import pickle
import numpy as np
import shap
import matplotlib.pyplot as plt

# Load artifacts
import joblib

pipeline = joblib.load("model.pkl")
tfidf = joblib.load("vectorizer.pkl")
explainer = joblib.load("explainer.pkl")

THRESHOLD = 0.35   # tuned threshold

st.set_page_config(page_title="Job Scam Risk Detector", layout="centered")

st.title("🔎 Explainable AI – Job Scam Risk Scoring System")

with st.container():
    job_text = st.text_area("Paste Job Description", height=250)

    col1, col2 = st.columns(2)
    with col1:
        salary_missing = st.checkbox("Salary Not Mentioned")
        telecommuting = st.checkbox("Remote Job")
    with col2:
        has_logo = st.checkbox("Company Logo Present", value=True)
        has_questions = st.checkbox("Screening Questions Present", value=True)

if st.button("Analyze Scam Risk"):
    if job_text.strip() == "":
        st.warning("Please enter job description.")
    else:
        input_data = {
            "combined_text": job_text,
            "salary_mean": 0,
            "salary_missing": int(salary_missing),
            "telecommuting": int(telecommuting),
            "has_company_logo": int(has_logo),
            "has_questions": int(has_questions)
        }

        import pandas as pd
        X_input = pd.DataFrame([input_data])

        prob = pipeline.predict_proba(X_input)[0][1]
        risk_score = round(prob * 100, 2)

        st.subheader("📊 Scam Risk Score")
        st.metric("Risk (%)", risk_score)

        if prob >= THRESHOLD:
            st.error("🚨 High Scam Risk")
        else:
            st.success("✅ Low Scam Risk")

        st.divider()
        st.subheader("🧠 Why this prediction?")

        import pandas as pd

        preprocessor = pipeline.named_steps["preprocessor"]
        classifier = pipeline.named_steps["classifier"]

        X_input_transformed = preprocessor.transform(X_input)

        background_df = pd.DataFrame({
            "combined_text": [
                "software engineer full time job",
                "work from home opportunity",
                "urgent hiring immediate joining",
                "salary discussed after interview"
            ],
            "salary_mean": [0, 0, 0, 0],
            "salary_missing": [0, 1, 0, 1],
            "telecommuting": [0, 1, 0, 0],
            "has_company_logo": [1, 0, 0, 1],
            "has_questions": [1, 0, 0, 0]
        })

        background_transformed = preprocessor.transform(background_df)

        explainer = shap.LinearExplainer(
            classifier,
            background_transformed,
            feature_perturbation="interventional"
        )

        shap_values = explainer(X_input_transformed)

        st.caption(
    "The chart below shows which words and signals increased or decreased the scam risk for this job posting."
)

        fig = plt.figure(figsize=(8, 5))
        shap.plots.waterfall(shap_values[0], max_display=10, show=False)
        st.pyplot(fig)
        plt.close(fig)

        st.caption("Built using Python, Scikit-learn, SHAP, and Streamlit")
       