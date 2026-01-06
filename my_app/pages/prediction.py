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
# LOAD DATA
# ===============================
@st.cache_data
def load_data():
    return pd.read_csv("my_app/breast_clean.csv")

data = load_data()

TARGET_COL = "diagnosis"
ID_COL = "id"

feature_columns = [c for c in data.columns if c not in [TARGET_COL, ID_COL]]

# ===============================
# SIDEBAR
# ===============================
st.sidebar.header("📊 Dataset Info")
st.sidebar.write(f"Features used: {len(feature_columns)}")
st.sidebar.dataframe(data[feature_columns].head())

# ===============================
# LOAD MODEL
# ===============================
st.header("📁 Load Model")

model_files = [f for f in os.listdir() if f.endswith(".pkl")]

selected_file = st.selectbox("Select model", model_files)

model = joblib.load(selected_file)

st.success(f"Model loaded: {selected_file}")
st.info(f"Model expects {model.n_features_in_} features")

# ===============================
# INPUT FORM
# ===============================
st.header("🧪 Patient Measurements")

inputs = {}

for col in feature_columns:
    inputs[col] = st.number_input(col, value=float(data[col].mean()))

input_df = pd.DataFrame([inputs])

st.subheader("Input Data")
st.dataframe(input_df)

# ===============================
# PREDICTION
# ===============================
st.header("🚀 Prediction")

if st.button("Predict"):
    try:
        prediction = model.predict(input_df)
        predicted_index = prediction[0]

        class_labels = model.classes_

        # 🔥 CORRECT LABEL MAPPING
        predicted_label = (
            class_labels[predicted_index]
            if isinstance(predicted_index, (int, np.integer))
            else predicted_index
        )

        if predicted_label in ["M", 1]:
            st.error("🔴 MALIGNANT (Cancer Detected)")
        else:
            st.success("🟢 BENIGN (No Cancer)")

        # Probabilities
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_df)
            proba_df = pd.DataFrame(proba, columns=class_labels)
            st.subheader("Prediction Probabilities")
            st.dataframe(proba_df)

    except Exception:
        st.error("Prediction failed")
        st.code(traceback.format_exc())

# ===============================
# FOOTER
# ===============================
st.markdown("---")
st.markdown(
    "Developed by **Mercy Mwova** | "
    "[GitHub](https://github.com/hug627) | "
    "[LinkedIn](https://www.linkedin.com/in/mercy-mwova-3b4b9821b/)"
)
