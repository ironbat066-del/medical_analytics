# ==========================================================
# Medical Rehabilitation Analytics Dashboard
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib

from datetime import datetime

# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Medical Rehabilitation Analytics",
    page_icon="🏥",
    layout="wide"
)

# ==========================================================
# LOAD MODELS
# ==========================================================

classifier_model = joblib.load("models/best_classifier.joblib")
forecast_model = joblib.load("models/best_forecasting_model.joblib")

label_encoders = joblib.load("models/label_encoders.joblib")
numerical_imputer = joblib.load("models/numerical_imputer.joblib")
categorical_imputer = joblib.load("models/categorical_imputer.joblib")

feature_importance = pd.read_csv(
    "models/feature_importance.csv"
)

forecast_importance = pd.read_csv(
    "models/forecast_feature_importance.csv"
)

st.sidebar.success("✅ Models Loaded Successfully")

# ==========================================================
# TITLE
# ==========================================================

st.title("🏥 Medical Rehabilitation Analytics")

st.markdown("""
### Machine Learning Dashboard

This dashboard provides

- 🤖 No Show Prediction
- 📈 Demand Forecast
- 📊 Model Insights

""")

# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Module",
    [
        "Home",
        "No Show Prediction",
        "Demand Forecast",
        "Model Insights"
    ]
)

# ==========================================================
# HOME PAGE
# ==========================================================

if page == "Home":

    st.header("🏠 Dashboard Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Classification Model",
            "Random Forest"
        )

    with col2:
        st.metric(
            "Forecast Model",
            "Linear Regression"
        )

    with col3:
        st.metric(
            "Project Status",
            "Completed"
        )

    st.success(
        "Medical Rehabilitation Analytics Dashboard Loaded Successfully."
    )

    st.markdown("---")

    st.subheader("📌 Project Objectives")

    st.write("""
- Predict patients likely to miss appointments.
- Forecast daily appointment demand.
- Identify important factors affecting patient attendance.
- Assist hospitals in planning staff and resources.
""")

    st.markdown("---")

    st.subheader("🛠 Technologies Used")

    tech1, tech2, tech3 = st.columns(3)

    with tech1:
        st.write("✔ Python")
        st.write("✔ Pandas")
        st.write("✔ NumPy")

    with tech2:
        st.write("✔ Scikit-Learn")
        st.write("✔ XGBoost")
        st.write("✔ Joblib")

    with tech3:
        st.write("✔ Streamlit")
        st.write("✔ Machine Learning")
        st.write("✔ Data Analytics")
# ==========================================================
# NO SHOW PREDICTION
# ==========================================================

if page == "No Show Prediction":

    st.header("🤖 No Show Prediction")

    st.write("Enter patient details below.")

    col1, col2 = st.columns(2)

    with col1:

        age = st.number_input(
            "Age",
            min_value=0,
            max_value=100,
            value=30
        )

        gender = st.selectbox(
            "Gender",
            ["F", "M", "I"]
        )

        specialty = st.selectbox(
            "Specialty",
            [
                "assist",
                "enf",
                "occupational therapy",
                "pedagogo",
                "physiotherapy",
                "psychotherapy",
                "sem especialidade",
                "speech therapy"
            ]
        )

        place = st.selectbox(
            "Place",
            list(label_encoders["place"].classes_)
        )

        appointment_shift = st.selectbox(
            "Appointment Shift",
            [
                "morning",
                "afternoon"
            ]
        )

        hypertension = st.selectbox(
            "Hypertension",
            [0, 1]
        )

        diabetes = st.selectbox(
            "Diabetes",
            [0, 1]
        )

        alcoholism = st.selectbox(
            "Alcoholism",
            [0, 1]
        )

    with col2:

        scholarship = st.selectbox(
            "Scholarship",
            [0, 1]
        )

        sms_received = st.selectbox(
            "SMS Received",
            [0, 1]
        )

        disability = st.selectbox(
            "Disability",
            [
                " ",
                "intellectual",
                "motor"
            ]
        )

        handcap = st.selectbox(
            "Handcap",
            [0, 1]
        )

        appointment_time = st.slider(
            "Appointment Hour",
            7,
            18,
            10
        )

        weather = st.selectbox(
            "Weather",
            [
                "Sunny",
                "Cloudy",
                "Rainy"
            ]
        )

    st.divider()

    if st.button("Predict No Show Risk"):

        under_12_years_old = int(age < 12)
        over_60_years_old = int(age >= 60)

        patient_needs_companion = int(
            under_12_years_old or over_60_years_old
        )

        needs_companion = patient_needs_companion

        if age < 12:
            age_group = "Child"
        elif age < 18:
            age_group = "Teen"
        elif age < 40:
            age_group = "Young Adult"
        elif age < 60:
            age_group = "Adult"
        else:
            age_group = "Senior"

        health_score = (
            hypertension +
            diabetes +
            alcoholism +
            handcap
        )

        is_rainy = int(weather == "Rainy")
        is_hot = int(weather == "Sunny")

        appointment_shift_map = {
            "morning": 0,
            "afternoon": 1
        }

        gender = label_encoders["gender"].transform([gender])[0]
        specialty = label_encoders["specialty"].transform([specialty])[0]
        place = label_encoders["place"].transform([place])[0]
        disability = label_encoders["disability"].transform([disability])[0]
        age_group = label_encoders["age_group"].transform([age_group])[0]

        input_data = pd.DataFrame({

            "specialty":[specialty],
            "appointment_time":[appointment_time],
            "gender":[gender],
            "disability":[disability],
            "place":[place],
            "appointment_shift":[appointment_shift_map[appointment_shift]],
            "age":[age],
            "under_12_years_old":[under_12_years_old],
            "over_60_years_old":[over_60_years_old],
            "patient_needs_companion":[patient_needs_companion],
            "average_temp_day":[30],
            "average_rain_day":[0],
            "max_temp_day":[32],
            "max_rain_day":[0],
            "rainy_day_before":[0],
            "storm_day_before":[0],
            "rain_intensity":[0],
            "heat_intensity":[1],
            "appointment_date_continuous":[0],
            "Hipertension":[hypertension],
            "Diabetes":[diabetes],
            "Alcoholism":[alcoholism],
            "Handcap":[handcap],
            "Scholarship":[scholarship],
            "SMS_received":[sms_received],
            "age_group":[age_group],
            "health_score":[health_score],
            "is_rainy":[is_rainy],
            "is_hot":[is_hot],
            "needs_companion":[needs_companion]

        })

        prediction = classifier_model.predict(input_data)[0]
        probability = classifier_model.predict_proba(input_data)[0][1]

        st.success("Prediction Completed Successfully!")

        if prediction == 1:
            st.error("⚠ High Risk of No Show")
        else:
            st.success("✅ Low Risk of No Show")

        st.metric(
            "Probability",
            f"{probability*100:.2f}%"
        )
# ==========================================================
# DEMAND FORECAST
# ==========================================================

if page == "Demand Forecast":

    st.header("📈 Demand Forecast")

    st.write("Predict expected daily appointments.")

    col1, col2, col3 = st.columns(3)

    with col1:
        year = st.number_input("Year", 2024, 2035, 2026)
        month = st.slider("Month", 1, 12, 6)

    with col2:
        day = st.slider("Day", 1, 31, 15)

        weekday = st.selectbox(
            "Weekday",
            [
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday"
            ]
        )

    with col3:
        week = st.slider("Week Number", 1, 53, 25)
        quarter = st.selectbox("Quarter", [1, 2, 3, 4])

    if st.button("Forecast Appointments"):

        weekday_map = {
            "Monday":0,
            "Tuesday":1,
            "Wednesday":2,
            "Thursday":3,
            "Friday":4,
            "Saturday":5,
            "Sunday":6
        }

        input_data = pd.DataFrame({

            "Year":[year],
            "Month":[month],
            "Day":[day],
            "Weekday":[weekday_map[weekday]],
            "Week":[week],
            "Quarter":[quarter]

        })

        prediction = forecast_model.predict(input_data)[0]
        prediction = max(0, round(prediction))

        st.success("Forecast Completed Successfully!")

        st.metric(
            "Expected Daily Appointments",
            f"{prediction} Patients"
        )

        st.write(
            "Forecast generated using the trained Linear Regression model."
        )

# ==========================================================
# MODEL INSIGHTS
# ==========================================================

if page == "Model Insights":

    st.header("📊 Model Insights")

    tab1, tab2 = st.tabs(
        [
            "Classification",
            "Forecasting"
        ]
    )

    with tab1:

        st.subheader("Top Classification Features")

        st.dataframe(
            feature_importance.head(15),
            use_container_width=True
        )

        st.bar_chart(
            feature_importance.set_index("Feature").head(10)
        )

        st.subheader("Classification Metrics")

        st.table(

            pd.DataFrame({

                "Metric":[
                    "Accuracy",
                    "Precision",
                    "Recall",
                    "F1 Score",
                    "ROC AUC"
                ],

                "Value":[
                    "70.30 %",
                    "52.40 %",
                    "71.40 %",
                    "60.40 %",
                    "76.60 %"
                ]

            })

        )

    with tab2:

        st.subheader("Top Forecast Features")

        st.dataframe(
            forecast_importance.head(15),
            use_container_width=True
        )

        st.bar_chart(
            forecast_importance.set_index("Feature").head(10)
        )

        st.subheader("Forecast Metrics")

        st.table(

            pd.DataFrame({

                "Metric":[
                    "RMSE",
                    "MAE",
                    "MAPE",
                    "R² Score"
                ],

                "Value":[
                    "284.83",
                    "229.72",
                    "4552.07",
                    "-0.0046"
                ]

            })

        )

# ==========================================================
# FOOTER
# ==========================================================

st.markdown("---")

st.caption(
    "Medical Rehabilitation Analytics Dashboard | Developed using Python, Streamlit, Scikit-Learn and Machine Learning"
)