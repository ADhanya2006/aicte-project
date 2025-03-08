import streamlit as st
import numpy as np
import pandas as pd
import time
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import percentileofscore
import joblib
st.set_page_config(page_title="Enhanced Fitness Tracker", page_icon="🏋", layout="wide")
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
st.title("🏋 Fitness Tracker")
st.write("Track fitness progress, set goals, and estimate calorie burn!")
st.sidebar.header("User Input ")
def user_input_features():
    age = st.sidebar.number_input("Age", 10, 100, 30)
    weight = st.sidebar.number_input("Weight (kg)", 30, 150, 70)
    height = st.sidebar.number_input("Height (cm)", 100, 220, 170)
    bmi = weight / ((height / 100) ** 2)
    duration = st.sidebar.slider("Exercise Duration (min)", 5, 120, 30)
    steps = st.sidebar.number_input("Steps Taken", 0, 20000, 5000)
    heart_rate = st.sidebar.slider("Heart Rate (bpm)", 50, 180, 90)
    body_temp = st.sidebar.slider("Body Temperature (°C)", 35.0, 42.0, 37.0)
    gender = st.sidebar.radio("Gender", ("Male", "Female"))
    activity_type = st.sidebar.selectbox("Activity Type", ["Walking", "Running", "Cycling", "Strength Training"])
    intensity = st.sidebar.selectbox("Exercise Intensity", ["Low", "Moderate", "High"])
    gender_encoded = 1 if gender == "Male" else 0
    activity_encoded = {"Walking": 1, "Running": 2, "Cycling": 3, "Strength Training": 4}[activity_type]
    intensity_encoded = {"Low": 1, "Moderate": 2, "High": 3}[intensity]
    return pd.DataFrame({
        "Gender_male": [gender_encoded],
        "Age": [age],
        "BMI": [bmi],
        "Duration": [duration],
        "Steps": [steps],
        "Heart_Rate": [heart_rate],
        "Body_Temp": [body_temp],
        "Activity": [activity_encoded],
        "Intensity": [intensity_encoded]
    })
df = user_input_features()
st.write("### Selected Parameters:")
st.dataframe(df)
np.random.seed(42)
data = pd.DataFrame({
    "Gender_male": np.random.randint(0, 2, 500),
    "Age": np.random.randint(18, 60, 500),
    "BMI": np.random.uniform(18.5, 35.0, 500),
    "Duration": np.random.randint(10, 120, 500),
    "Steps": np.random.randint(1000, 20000, 500),
    "Heart_Rate": np.random.randint(60, 160, 500),
    "Body_Temp": np.random.uniform(36.0, 40.0, 500),
    "Activity": np.random.randint(1, 5, 500),  # Activity levels
    "Intensity": np.random.randint(1, 4, 500),
})

data["Calories"] = (
    4 * data["Duration"] +
    0.05 * data["Steps"] +
    2 * data["Heart_Rate"] +
    3 * data["BMI"] +
    10 * data["Activity"] +
    5 * data["Intensity"] +
    np.random.normal(0, 30, 500)  
)
X = data.drop("Calories", axis=1)
y = data["Calories"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model = GradientBoostingRegressor(n_estimators=150, learning_rate=0.1, random_state=42)
model.fit(X_train_scaled, y_train)
joblib.dump(model, "fitness_model.pkl")
joblib.dump(scaler, "scaler.pkl")
df_scaled = scaler.transform(df[X_train.columns])
prediction = model.predict(df_scaled)[0]
st.write("### Predicted Calories Burned:")
st.metric(label="Estimated Calories Burned", value=f"{round(prediction, 2)} kcal")
st.sidebar.header("Set Your Goal")
goal = st.sidebar.number_input("Weekly Calorie Burn Goal", 1000, 10000, 3500)
if "history" not in st.session_state:
    st.session_state["history"] = []
st.session_state["history"].append({"Calories Burned": round(prediction, 2), "Timestamp": time.strftime("%H:%M:%S")})
history_df = pd.DataFrame(st.session_state["history"])
total_calories_burned = history_df["Calories Burned"].sum()
progress = min(total_calories_burned / goal, 1.0)
st.sidebar.progress(progress)
st.sidebar.write(f"**Goal Progress: {round(progress * 100, 1)}%**")
if len(history_df) > 1:
    st.write("### Progress Over Time")
    st.line_chart(history_df.set_index("Timestamp"))
if st.sidebar.button("Reset Progress"):
    st.session_state["history"] = []
    st.rerun()
