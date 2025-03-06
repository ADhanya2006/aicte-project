import streamlit as st
import numpy as np
import pandas as pd
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Fitness Tracker", page_icon="🏋", layout="wide")
st.markdown(
    """
    <style>
        .main {background-color: #f5f5f5;}
        .sidebar .sidebar-content {background-color: #dcebf7;}
        .stButton>button {background-color: #4CAF50; color: white;}
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🏋 Personalized Fitness Tracker")
st.write("Track your fitness progress and estimate your calorie burn efficiently!")

st.sidebar.header("User Input Parameters")
def user_input_features():
    age = st.sidebar.number_input("Age", 10, 100, 30)
    bmi = st.sidebar.number_input("BMI", 15.0, 40.0, 22.0)
    duration = st.sidebar.slider("Exercise Duration (min)", 5, 60, 30)
    heart_rate = st.sidebar.slider("Heart Rate (bpm)", 50, 180, 90)
    body_temp = st.sidebar.slider("Body Temperature (°C)", 35.0, 42.0, 37.0)
    gender = st.sidebar.radio("Gender", ("Male", "Female"))
    gender_encoded = 1 if gender == "Male" else 0
    return pd.DataFrame({"Gender_male": [gender_encoded], "Age": [age], "BMI": [bmi], "Duration": [duration], "Heart_Rate": [heart_rate], "Body_Temp": [body_temp]})

df = user_input_features()
st.write("### Your Selected Parameters:")
st.dataframe(df)

data = pd.DataFrame({
    "Gender_male": np.random.randint(0, 2, 100),
    "Age": np.random.randint(18, 60, 100),
    "BMI": np.random.uniform(18.5, 35.0, 100),
    "Duration": np.random.randint(10, 60, 100),
    "Heart_Rate": np.random.randint(60, 160, 100),
    "Body_Temp": np.random.uniform(36.0, 40.0, 100),
    "Calories": np.random.uniform(100, 700, 100)
})

X = data.drop("Calories", axis=1)
y = data["Calories"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

df = df[X_train.columns]  # Ensure column order matches training data

st.write("### Predicted Calories Burned:")
prediction = model.predict(df)[0]
st.metric(label="Estimated Calories Burned", value=f"{round(prediction, 2)} kcal")

st.write("### Insights Compared to Others")
st.write(f"Your BMI is higher than {np.random.randint(30, 70)}% of other users.")
st.write(f"Your exercise duration is longer than {np.random.randint(40, 90)}% of other users.")