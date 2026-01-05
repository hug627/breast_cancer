import pandas as pd
import streamlit as st
import numpy as np
import joblib
import os
import traceback

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(page_title="Breast Cancer Prediction", layout="wide")
st.title("🤖 Breast Cancer Prediction System")

# ===============================
# LOAD DATA (FOR FEATURE ORDER)
# ===============================
@st.cache_data
def load_data():
    try:
        csv_file = "my_app/Breast_cancer_dataset (1).csv"
        return pd.read_csv(csv_file)
    except Exception as e:
        st.error(f"Cannot load dataset: {e}")
        return None

data = load_data()
if data is None:
    st.stop()

# Target column must NOT be included
TARGET_COL = "diagnosis"
ID_COL = "id"

feature_columns = [col for col in data.columns if col not in [TARGET_COL, ID_COL]]

# ===============================
# SIDEBAR INFO
# ===============================
st.sidebar.header("📊 Dataset Info")
st.sidebar.write(f"Total Features Used: {len(feature_columns)}")
st.sidebar.write(feature_columns)
st.sidebar.subheader("Sample Data")
st.sidebar.dataframe(data[feature_columns].head(3))

# ===============================
# LOAD MODEL
# ===============================
st.header("📁 Load Model")

model_files = [f for f in os.listdir('.') if f.endswith('.pkl')]

if not model_files:
    st.error("❌ No .pkl model files found in this directory.")
    st.stop()

selected_file = st.selectbox("Select model file", model_files)

try:
    model = joblib.load(selected_file)
    st.success(f"✅ Model Loaded: {selected_file}")

    if hasattr(model, "n_features_in_"):
        st.info(f"Model expects {model.n_features_in_} features")
        if model.n_features_in_ != len(feature_columns):
            st.error(
                f"⚠ Feature mismatch: Model expects {model.n_features_in_}, "
                f"but dataset has {len(feature_columns)}"
            )
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# ===============================
# INPUT FORM
# ===============================
st.header("🧪 Enter Patient Measurements")

col1, col2, col3 = st.columns(3)

with col1:
    radius_mean = st.number_input("Radius Mean", 0.0, 100.0, 14.0, 0.1)
    texture_mean = st.number_input("Texture Mean", 0.0, 100.0, 20.0, 0.1)
    perimeter_mean = st.number_input("Perimeter Mean", 0.0, 200.0, 90.0, 0.1)
    area_mean = st.number_input("Area Mean", 0.0, 2000.0, 600.0, 1.0)
    smoothness_mean = st.number_input("Smoothness Mean", 0.0, 1.0, 0.1, 0.001)
    compactness_mean = st.number_input("Compactness Mean", 0.0, 1.0, 0.1, 0.001)
    concavity_mean = st.number_input("Concavity Mean", 0.0, 1.0, 0.1, 0.001)
    concave_points_mean = st.number_input("Concave Points Mean", 0.0, 1.0, 0.05, 0.001)
    symmetry_mean = st.number_input("Symmetry Mean", 0.0, 1.0, 0.2, 0.001)
    fractal_dimension_mean = st.number_input("Fractal Dimension Mean", 0.0, 1.0, 0.06, 0.001)

with col2:
    radius_se = st.number_input("Radius SE", 0.0, 10.0, 0.3, 0.01)
    texture_se = st.number_input("Texture SE", 0.0, 10.0, 1.0, 0.01)
    perimeter_se = st.number_input("Perimeter SE", 0.0, 20.0, 2.0, 0.01)
    area_se = st.number_input("Area SE", 0.0, 200.0, 20.0, 0.1)
    smoothness_se = st.number_input("Smoothness SE", 0.0, 1.0, 0.01, 0.001)
    compactness_se = st.number_input("Compactness SE", 0.0, 1.0, 0.02, 0.001)
    concavity_se = st.number_input("Concavity SE", 0.0, 1.0, 0.03, 0.001)
    concave_points_se = st.number_input("Concave Points SE", 0.0, 1.0, 0.01, 0.001)
    symmetry_se = st.number_input("Symmetry SE", 0.0, 1.0, 0.02, 0.001)
    fractal_dimension_se = st.number_input("Fractal Dimension SE", 0.0, 1.0, 0.003, 0.001)

with col3:
    radius_worst = st.number_input("Radius Worst", 0.0, 100.0, 16.0, 0.1)
    texture_worst = st.number_input("Texture Worst", 0.0, 100.0, 25.0, 0.1)
    perimeter_worst = st.number_input("Perimeter Worst", 0.0, 200.0, 110.0, 0.1)
    area_worst = st.number_input("Area Worst", 0.0, 3000.0, 800.0, 1.0)
    smoothness_worst = st.number_input("Smoothness Worst", 0.0, 1.0, 0.15, 0.001)
    compactness_worst = st.number_input("Compactness Worst", 0.0, 1.0, 0.2, 0.001)
    concavity_worst = st.number_input("Concavity Worst", 0.0, 1.0, 0.3, 0.001)
    concave_points_worst = st.number_input("Concave Points Worst", 0.0, 1.0, 0.1, 0.001)
    symmetry_worst = st.number_input("Symmetry Worst", 0.0, 1.0, 0.3, 0.001)
    fractal_dimension_worst = st.number_input("Fractal Dimension Worst", 0.0, 1.0, 0.08, 0.001)

# ===============================
# CREATE INPUT DATAFRAME
# ===============================
input_data = {
    col: locals()[col] for col in feature_columns
}

input_df = pd.DataFrame([input_data])

# Enforce correct feature order
input_df = input_df[feature_columns]

st.subheader("📋 Input Patient Data")
st.dataframe(input_df)

# ===============================
# PREDICTION
# ===============================
st.header("🚀 Prediction Result")

if st.button("Predict"):
    try:
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)

        predicted_class = prediction[0]

        if predicted_class == "B":
            st.success("🟢 Prediction: Benign (No Cancer)")
        else:
            st.error("🔴 Prediction: Malignant (Cancer Detected)")

        confidence = np.max(prediction_proba) * 100
        st.info(f"Confidence Level: {confidence:.2f}%")

        proba_df = pd.DataFrame(
            prediction_proba,
            columns=model.classes_
        )
        st.subheader("Prediction Probabilities")
        st.dataframe(proba_df)

    except Exception as e:
        st.error("Prediction failed")
        st.text(traceback.format_exc())

# ===============================
# FOOTER
# ===============================
st.markdown("---")
st.markdown(
    "Developed by **Mercy Mwova** | "
    "[GitHub](https://github.com/hug627) | "
    "[LinkedIn](https://www.linkedin.com/in/mercy-mwova-3b4b9821b/)"
)
