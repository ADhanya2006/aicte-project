import streamlit as st
import numpy as np
import pandas as pd
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import percentileofscore

# Set page config
st.set_page_config(page_title="Fitness Tracker", page_icon="🏋", layout="wide")

# Custom Styling
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

# Sidebar Inputs
st.sidebar.header("User Input Parameters")
def user_input_features():
    age = st.sidebar.number_input("Age", 10, 100, 30)
    weight = st.sidebar.number_input("Weight (kg)", 30, 150, 70)
    height = st.sidebar.number_input("Height (cm)", 100, 220, 170)
    bmi = weight / ((height / 100) ** 2)
    duration = st.sidebar.slider("Exercise Duration (min)", 5, 120, 30)
    heart_rate = st.sidebar.slider("Heart Rate (bpm)", 50, 180, 90)
    body_temp = st.sidebar.slider("Body Temperature (°C)", 35.0, 42.0, 37.0)
    gender = st.sidebar.radio("Gender", ("Male", "Female"))
    activity_type = st.sidebar.selectbox("Activity Type", ["Walking", "Running", "Cycling", "Strength Training"])
    
    # Encode categorical features
    gender_encoded = 1 if gender == "Male" else 0
    activity_encoded = {"Walking": 1, "Running": 2, "Cycling": 3, "Strength Training": 4}[activity_type]

    return pd.DataFrame({
        "Gender_male": [gender_encoded],
        "Age": [age],
        "BMI": [bmi],
        "Duration": [duration],
        "Heart_Rate": [heart_rate],
        "Body_Temp": [body_temp],
        "Activity": [activity_encoded]
    })

df = user_input_features()
st.write("### Your Selected Parameters:")
st.dataframe(df)

# Generate Realistic Data for Model Training
np.random.seed(42)
data = pd.DataFrame({
    "Gender_male": np.random.randint(0, 2, 500),
    "Age": np.random.randint(18, 60, 500),
    "BMI": np.random.uniform(18.5, 35.0, 500),
    "Duration": np.random.randint(10, 120, 500),
    "Heart_Rate": np.random.randint(60, 160, 500),
    "Body_Temp": np.random.uniform(36.0, 40.0, 500),
    "Activity": np.random.randint(1, 5, 500),  # Activity levels: Walking-1, Running-2, Cycling-3, Strength Training-4
})

# Simulated calorie burn based on basic metabolic formula
data["Calories"] = (
    5 * data["Duration"] +
    2 * data["Heart_Rate"] +
    3 * data["BMI"] +
    10 * data["Activity"] +
    np.random.normal(0, 30, 500)  # Adding slight randomness
)

# Train Model
X = data.drop("Calories", axis=1)
y = data["Calories"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Make Prediction
df_scaled = scaler.transform(df[X_train.columns])
prediction = model.predict(df_scaled)[0]

st.write("### Predicted Calories Burned:")
st.metric(label="Estimated Calories Burned", value=f"{round(prediction, 2)} kcal")

# Insights Based on Percentiles
bmi_percentile = percentileofscore(data["BMI"], df["BMI"][0])
duration_percentile = percentileofscore(data["Duration"], df["Duration"][0])

st.write("### Insights Compared to Others")
st.write(f"Your BMI is higher than **{round(bmi_percentile, 1)}%** of other users.")
st.write(f"Your exercise duration is longer than **{round(duration_percentile, 1)}%** of other users.")

# Store Session Data for Tracking
if "history" not in st.session_state:
    st.session_state["history"] = []

st.session_state["history"].append({"Calories Burned": round(prediction, 2), "Timestamp": time.strftime("%H:%M:%S")})

# Convert to DataFrame for Visualization
history_df = pd.DataFrame(st.session_state["history"])

# Display Tracking Graph
if len(history_df) > 1:
    st.write("### Progress Over Time")
    st.line_chart(history_df.set_index("Timestamp"))