# streamlit_app.py

import streamlit as st
import numpy as np
import tensorflow as tf
import pickle

# =========================
# LOAD MODEL
# =========================
model = tf.keras.models.load_model("breast_cancer_model.keras")

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Breast Cancer Prediction",

    layout="centered"
)

st.title(" Breast Cancer Prediction App")
st.write("Enter the values below and click Predict")

# =========================
# INPUT FIELDS
# =========================

mean_radius = st.number_input("Mean Radius", value=14.0)
mean_texture = st.number_input("Mean Texture", value=20.0)
mean_perimeter = st.number_input("Mean Perimeter", value=90.0)
mean_area = st.number_input("Mean Area", value=600.0)
mean_smoothness = st.number_input("Mean Smoothness", value=0.1)
mean_compactness = st.number_input("Mean Compactness", value=0.1)
mean_concavity = st.number_input("Mean Concavity", value=0.1)
mean_concave_points = st.number_input("Mean Concave Points", value=0.05)
mean_symmetry = st.number_input("Mean Symmetry", value=0.2)
mean_fractal_dimension = st.number_input("Mean Fractal Dimension", value=0.06)

radius_error = st.number_input("Radius Error", value=0.5)
texture_error = st.number_input("Texture Error", value=1.0)
perimeter_error = st.number_input("Perimeter Error", value=3.0)
area_error = st.number_input("Area Error", value=40.0)
smoothness_error = st.number_input("Smoothness Error", value=0.005)
compactness_error = st.number_input("Compactness Error", value=0.02)
concavity_error = st.number_input("Concavity Error", value=0.03)
concave_points_error = st.number_input("Concave Points Error", value=0.01)
symmetry_error = st.number_input("Symmetry Error", value=0.02)
fractal_dimension_error = st.number_input("Fractal Dimension Error", value=0.003)

worst_radius = st.number_input("Worst Radius", value=16.0)
worst_texture = st.number_input("Worst Texture", value=25.0)
worst_perimeter = st.number_input("Worst Perimeter", value=110.0)
worst_area = st.number_input("Worst Area", value=800.0)
worst_smoothness = st.number_input("Worst Smoothness", value=0.14)
worst_compactness = st.number_input("Worst Compactness", value=0.25)
worst_concavity = st.number_input("Worst Concavity", value=0.3)
worst_concave_points = st.number_input("Worst Concave Points", value=0.12)
worst_symmetry = st.number_input("Worst Symmetry", value=0.3)
worst_fractal_dimension = st.number_input("Worst Fractal Dimension", value=0.08)

# =========================
# PREDICTION BUTTON
# =========================

if st.button("Predict"):

    input_data = np.array([[
        mean_radius,
        mean_texture,
        mean_perimeter,
        mean_area,
        mean_smoothness,
        mean_compactness,
        mean_concavity,
        mean_concave_points,
        mean_symmetry,
        mean_fractal_dimension,
        radius_error,
        texture_error,
        perimeter_error,
        area_error,
        smoothness_error,
        compactness_error,
        concavity_error,
        concave_points_error,
        symmetry_error,
        fractal_dimension_error,
        worst_radius,
        worst_texture,
        worst_perimeter,
        worst_area,
        worst_smoothness,
        worst_compactness,
        worst_concavity,
        worst_concave_points,
        worst_symmetry,
        worst_fractal_dimension
    ]])

    prediction = model.predict(input_data)

    predicted_value = prediction[0][0]

    st.subheader("Prediction Score")
    st.write(predicted_value)

    if predicted_value > 0.5:
        st.error(" Malignant Cancer Detected")
    else:
        st.success(" Benign Cancer Detected")
       
       
       
    # Step -1 :=    
    # Create a Environment :  python -m venv venv
   
    # Step - 2 :=
    # Activate Environment in cmd  : Activate Environment
   
    # step - 3 :=
    # Install Libraries : pip install tensorflow streamlit numpy

