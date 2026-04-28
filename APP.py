import streamlit as st
st.set_page_config(page_title="LDH Conservative Treatment Failure Predictor", layout="centered")

import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# ------------------------------
# 1. Feature mappings
# ------------------------------
categorical_mappings = {
    "Herniation_sagittal": {
        "0: No herniation": 0,
        "1: Protrusion": 1,
        "2: Extrusion": 2,
        "3: Sequestration": 3,
    },
    "Modic_grade": {
        "0: None": 0,
        "1: Type I": 1,
        "2: Type II": 2,
        "3: Type III": 3,
    },
    "Pfirrmann_grade": {
        "1: Grade I (normal)": 1,
        "2: Grade II": 2,
        "3: Grade III": 3,
        "4: Grade IV": 4,
        "5: Grade V (severe)": 5,
    },
}

feature_ranges = {
    "Age": {"type": "numerical", "min": 18.0, "max": 80.0, "default": 56.0},
    "BMI": {"type": "numerical", "min": 10.0, "max": 50.0, "default": 24.5},
    "Lowback_vas": {"type": "numerical", "min": 0.0, "max": 10.0, "default": 5.0},
    "Leg_vas": {"type": "numerical", "min": 0.0, "max": 10.0, "default": 6.0},
    "Duration": {"type": "numerical", "min": 0.0, "max": 360.0, "default": 30.0},
    "Herniation_sagittal": {"type": "categorical", "mapping": categorical_mappings["Herniation_sagittal"]},
    "Modic_grade": {"type": "categorical", "mapping": categorical_mappings["Modic_grade"]},
    "Pfirrmann_grade": {"type": "categorical", "mapping": categorical_mappings["Pfirrmann_grade"]},
}

feature_order = ["Age", "BMI", "Lowback_vas", "Leg_vas", "Duration",
                 "Herniation_sagittal", "Modic_grade", "Pfirrmann_grade"]

# ------------------------------
# 2. Load resources (using numpy arrays to avoid feature name issues)
# ------------------------------
def init_resources():
    if 'model' not in st.session_state:
        with st.spinner("Loading SVM model..."):
            st.session_state.model = joblib.load('SVM.pkl')
    if 'scaler' not in st.session_state:
        with st.spinner("Loading feature scaler..."):
            st.session_state.scaler = joblib.load('scaler.pkl')
    if 'background_df_scaled' not in st.session_state:
        with st.spinner("Preparing background data for SHAP..."):
            try:
                bg_raw = pd.read_csv('background_sample.csv')
                bg_raw = bg_raw[feature_order]          # ensure correct order
                # Convert to numpy array to avoid column name checking
                bg_raw_array = bg_raw.values
                bg_scaled = st.session_state.scaler.transform(bg_raw_array)
                # Store scaled array as numpy, and also keep as DataFrame for SHAP (but will use array internally)
                st.session_state.background_scaled_array = bg_scaled
            except FileNotFoundError:
                st.warning("background_sample.csv not found. Using randomly generated background (SHAP may be less accurate).")
                np.random.seed(42)
                num_bg = 100
                bg_data = []
                for feat in feature_order:
                    info = feature_ranges[feat]
                    if info["type"] == "numerical":
                        vals = np.random.uniform(info["min"], info["max"], num_bg)
                    else:
                        vals = np.random.choice(list(info["mapping"].values()), num_bg)
                    bg_data.append(vals)
                bg_raw_array = np.array(bg_data).T
                bg_scaled = st.session_state.scaler.transform(bg_raw_array)
                st.session_state.background_scaled_array = bg_scaled
    if 'explainer' not in st.session_state:
        with st.spinner("Initializing SHAP KernelExplainer (first run may be slow)..."):
            # SHAP expects the background data as a matrix (numpy array)
            st.session_state.explainer = shap.KernelExplainer(
                st.session_state.model.predict_proba,
                st.session_state.background_scaled_array
            )

init_resources()
model = st.session_state.model
scaler = st.session_state.scaler
explainer = st.session_state.explainer

# ------------------------------
# 3. UI layout
# ------------------------------
st.title("🔮 Lumbar Disc Herniation: Prediction of Conservative Treatment Failure")
st.markdown("Enter patient clinical and imaging features. The model predicts the probability of **conservative treatment failure** (i.e., need for surgery or persistent pain).")

left_col, right_col = st.columns([0.35, 0.65])

with left_col:
    st.header("📋 Input Features")
    input_values = {}
    for feat in feature_order:
        info = feature_ranges[feat]
        if info["type"] == "numerical":
            val = st.number_input(
                label=feat,
                min_value=float(info["min"]),
                max_value=float(info["max"]),
                value=float(info["default"]),
                step=0.1 if feat in ["Lowback_vas", "Leg_vas", "BMI"] else 1.0,
                help=f"Range: {info['min']} - {info['max']}"
            )
            input_values[feat] = val
        else:
            display_opts = list(info["mapping"].keys())
            selected = st.selectbox(label=feat, options=display_opts, help="Select grade")
            input_values[feat] = info["mapping"][selected]
    predict_button = st.button("🔍 Predict", type="primary", use_container_width=True)

with right_col:
    if predict_button:
        # Create numpy array with correct order
        features_raw = np.array([[input_values[feat] for feat in feature_order]], dtype=float)
        # Scale using numpy array (no column names)
        features_scaled = scaler.transform(features_raw)
        
        proba = model.predict_proba(features_scaled)[0]
        # Assume classes: [success, failure]
        failure_prob = proba[1] * 100
        
        st.subheader("📊 Prediction Result")
        st.metric(
            label="Probability of Conservative Treatment Failure",
            value=f"{failure_prob:.1f}%",
            delta="Surgical intervention advised" if failure_prob > 75 else "Conservative treatment can continue"
        )
        
        with st.spinner("Computing feature contributions..."):
            # SHAP expects the sample as a matrix (numpy array)
            shap_values_all = explainer.shap_values(features_scaled)
            if len(shap_values_all.shape) == 3:
                shap_values_failure = shap_values_all[0, :, 1]
            else:
                shap_values_failure = shap_values_all[0, :]
            
            # For plots, we need feature names; create a DataFrame with scaled values for visualization
            scaled_df = pd.DataFrame(features_scaled, columns=feature_order)
            
            # Bar plot
            st.subheader("📌 Feature Contribution Magnitude")
            fig, ax = plt.subplots(figsize=(8, 4))
            shap.bar_plot(shap_values_failure, feature_names=feature_order, show=False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            # Dot plot (single sample)
            st.subheader("📌 Direction of Feature Contribution")
            fig, ax = plt.subplots(figsize=(8, 4))
            shap.summary_plot(
                shap_values_failure.reshape(1, -1),
                scaled_df,
                plot_type="dot",
                show=False
            )
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        st.info(
            f"**Clinical Interpretation**: Estimated failure probability = {failure_prob:.1f}%. " +
            ("Early surgical consultation may be considered." if failure_prob > 75 else "Standard conservative therapy is recommended with regular follow-up.")
        )
    else:
        st.info("👈 Please enter feature values on the left and click 'Predict'.")