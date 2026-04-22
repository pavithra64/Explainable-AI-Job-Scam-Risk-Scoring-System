import streamlit as st
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="Job Scam Risk Detector",
    page_icon="🔎",
    layout="centered"
)

# -------------------------------
# LOAD MODEL (CACHED)
# -------------------------------
@st.cache_resource
def load_model():
    pipeline = joblib.load("model.pkl")
    return pipeline

pipeline = load_model()

THRESHOLD = 0.35

# -------------------------------
# UI
# -------------------------------
st.title("🔎 Explainable AI – Job Scam Risk Scoring System")

st.write(
    "Analyze job descriptions and detect potential scam risks using Machine Learning + Explainable AI."
)

job_text = st.text_area("📄 Paste Job Description", height=250)

col1, col2 = st.columns(2)

with col1:
    salary_missing = st.checkbox("Salary Not Mentioned")
    telecommuting = st.checkbox("Remote Job")

with col2:
    has_logo = st.checkbox("Company Logo Present", value=True)
    has_questions = st.checkbox("Screening Questions Present", value=True)

# -------------------------------
# ANALYSIS
# -------------------------------
if st.button("🚀 Analyze Scam Risk"):

    if job_text.strip() == "":
        st.warning("⚠️ Please enter a job description.")
        st.stop()

    # Input Data
    input_data = {
        "combined_text": job_text,
        "salary_mean": 0,
        "salary_missing": int(salary_missing),
        "telecommuting": int(telecommuting),
        "has_company_logo": int(has_logo),
        "has_questions": int(has_questions)
    }

    X_input = pd.DataFrame([input_data])

    # Prediction
    prob = pipeline.predict_proba(X_input)[0][1]
    risk_score = round(prob * 100, 2)

    # -------------------------------
    # OUTPUT
    # -------------------------------
    st.subheader("📊 Scam Risk Score")
    st.metric("Risk (%)", risk_score)

    if prob >= THRESHOLD:
        st.error("🚨 High Scam Risk Detected")
    else:
        st.success("✅ Low Scam Risk")

    st.divider()

    # -------------------------------
    # EXPLAINABILITY (SHAP)
    # -------------------------------
    st.subheader("🧠 Why this prediction?")

    try:
        preprocessor = pipeline.named_steps["preprocessor"]
        classifier = pipeline.named_steps["classifier"]

        X_transformed = preprocessor.transform(X_input)

        # Background data (lightweight)
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

        @st.cache_resource
        def get_explainer(model, data):
            return shap.Explainer(model, data)

        explainer = get_explainer(classifier, background_transformed)
        shap_values = explainer(X_transformed)

        st.caption(
            "This chart shows which features increased or decreased the scam risk."
        )

        fig = plt.figure(figsize=(8, 5))
        shap.plots.waterfall(shap_values[0], max_display=10, show=False)
        st.pyplot(fig)
        plt.close(fig)

    except Exception as e:
        st.warning("⚠️ Explanation could not be generated.")
        st.text(str(e))

# -------------------------------
# FOOTER
# -------------------------------
st.divider()
st.caption("Built using Python, Scikit-learn, SHAP, and Streamlit")
