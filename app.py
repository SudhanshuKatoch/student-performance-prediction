import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import os
import altair as alt

# Load models and scaler
model_path = r"C:\student-performance-prediction\models\nn_model.h5"
scaler_path = r"C:\student-performance-prediction\models\scaler.pkl"

if not os.path.exists(model_path) or not os.path.exists(scaler_path):
    raise FileNotFoundError("Model or scaler not found. Please make sure they are trained and saved.")

model = load_model(model_path)
scaler = joblib.load(scaler_path)

# App configuration
st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓", layout="wide")

background_url = "https://your-professional-background-image-url.com"
st.markdown(
    f"""
    <style>
        .stApp {{
            background-image: url({background_url});
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        .main {{
            background: rgba(255, 255, 255, 0.8);
            padding: 20px;
            border-radius: 10px;
        }}
        .sidebar .sidebar-content {{
            background: rgba(255, 255, 255, 0.8);
            padding: 10px;
            border-radius: 10px;
        }}
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🎓 Student Performance Predictor")
st.write("An AI-powered tool to predict student exam performance.")

# Input form
st.sidebar.header("Input Student Details")
hours_studied = st.sidebar.slider("Hours Studied", 0, 40, 20)
attendance = st.sidebar.slider("Attendance (%)", 0, 100, 80)
parental_involvement = st.sidebar.selectbox("Parental Involvement", ["Low", "Medium", "High"])
access_to_resources = st.sidebar.selectbox("Access to Resources", ["Low", "Medium", "High"])
extracurricular_activities = st.sidebar.selectbox("Extracurricular Activities", ["No", "Yes"])
sleep_hours = st.sidebar.slider("Sleep Hours", 0, 10, 7)
previous_scores = st.sidebar.slider("Previous Scores", 0, 100, 70)
motivation_level = st.sidebar.selectbox("Motivation Level", ["Low", "Medium", "High"])
internet_access = st.sidebar.selectbox("Internet Access", ["No", "Yes"])
tutoring_sessions = st.sidebar.slider("Tutoring Sessions", 0, 10, 2)
family_income = st.sidebar.selectbox("Family Income", ["Low", "Medium", "High"])
teacher_quality = st.sidebar.selectbox("Teacher Quality", ["Low", "Medium", "High"])
school_type = st.sidebar.selectbox("School Type", ["Public", "Private"])
peer_influence = st.sidebar.selectbox("Peer Influence", ["Negative", "Neutral", "Positive"])
physical_activity = st.sidebar.slider("Physical Activity (Hours)", 0, 10, 4)
learning_disabilities = st.sidebar.selectbox("Learning Disabilities", ["No", "Yes"])
parental_education_level = st.sidebar.selectbox("Parental Education Level", ["High School", "College", "Postgraduate"])
distance_from_home = st.sidebar.selectbox("Distance from Home", ["Near", "Moderate", "Far"])
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])

# Process inputs
input_data = pd.DataFrame({
    "Hours_Studied": [hours_studied],
    "Attendance": [attendance],
    "Parental_Involvement": [parental_involvement],
    "Access_to_Resources": [access_to_resources],
    "Extracurricular_Activities": [extracurricular_activities],
    "Sleep_Hours": [sleep_hours],
    "Previous_Scores": [previous_scores],
    "Motivation_Level": [motivation_level],
    "Internet_Access": [internet_access],
    "Tutoring_Sessions": [tutoring_sessions],
    "Family_Income": [family_income],
    "Teacher_Quality": [teacher_quality],
    "School_Type": [school_type],
    "Peer_Influence": [peer_influence],
    "Physical_Activity": [physical_activity],
    "Learning_Disabilities": [learning_disabilities],
    "Parental_Education_Level": [parental_education_level],
    "Distance_from_Home": [distance_from_home],
    "Gender": [gender]
})

# One-hot encode input
input_data = pd.get_dummies(input_data, drop_first=True)
missing_cols = set(scaler.feature_names_in_) - set(input_data.columns)
for col in missing_cols:
    input_data[col] = 0
input_data = input_data[scaler.feature_names_in_]

# Add predict button below title
if st.button("Predict"):
    # Scale input
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)[0][0]
    predicted_score = round(prediction, 2)

    # Ensure prediction is within realistic bounds
    min_score, max_score = 0, 100
    final_score = min(max(predicted_score, min_score), max_score)

    st.subheader(f"Predicted Exam Score: {final_score:.2f}")

    # Visualize inputs
    st.write("### Input Data")
    st.write(input_data)

    # Improved visualization
    chart_data = pd.DataFrame(input_data.values[0], index=input_data.columns, columns=["Value"])
    chart = alt.Chart(chart_data.reset_index()).mark_bar().encode(
        x=alt.X('index', title='Feature'),
        y=alt.Y('Value', title='Value'),
        color=alt.condition(
            alt.datum.Value > 0,  # Condition for color
            alt.value('steelblue'),  # If true
            alt.value('lightgray')  # If false
        ),
        tooltip=['index', 'Value']
    ).properties(
        width=800,
        height=300,
        title="Input Features Visualization"
    ).interactive()
    st.altair_chart(chart)
else:
    st.write("Please use the sidebar to input details and click 'Predict'.")
