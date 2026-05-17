import streamlit as st
import pandas as pd
import joblib
import pyttsx3

# -----------------------------
# PAGE CONFIG
# -----------------------------

st.set_page_config(
    page_title="Accident Severity Prediction",
    page_icon="🚑",
    layout="wide"
)

# -----------------------------
# CUSTOM CSS
# -----------------------------

st.markdown("""
<style>

.main {
    background-color: #0f172a;
}

h1, h2, h3, h4 {
    color: white;
}

.stButton>button {
    background-color: #2563eb;
    color: white;
    border-radius: 10px;
    height: 50px;
    width: 100%;
    font-size: 18px;
    font-weight: bold;
}

.stButton>button:hover {
    background-color: #1d4ed8;
    color: white;
}

.prediction-box {
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    font-size: 28px;
    font-weight: bold;
    color: white;
    margin-top: 20px;
}

.high-risk {
    background-color: #dc2626;
}

.medium-risk {
    background-color: #f59e0b;
}

.low-risk {
    background-color: #16a34a;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# LOAD MODEL FILES
# -----------------------------

model = joblib.load("accident_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")
target_encoder = joblib.load("target_encoder.pkl")

# -----------------------------
# TITLE
# -----------------------------

st.title("🚑 Road Accident Severity Prediction System")

st.markdown("""
Predict accident severity using Machine Learning based on:

- Weather conditions
- Road conditions
- Speed limit
- Driver information
- Traffic environment
""")

# -----------------------------
# SIDEBAR
# -----------------------------

st.sidebar.header("📊 Project Information")

st.sidebar.info("""
AI-powered accident severity prediction system using Random Forest Machine Learning model.
""")

# -----------------------------
# INPUT FORM
# -----------------------------

col1, col2 = st.columns(2)

with col1:

    weather = st.selectbox(
        "🌧 Weather Condition",
        ['Rainy','Stormy', 'Clear', 'Foggy', 'Hazy']
    )

    road_type = st.selectbox(
        "🛣 Road Type",
        ['State Highway', 'Village Road', 'Urban Road' ,'National Highway']
    )

    road_condition= st.selectbox(
        "⚠ Road Condition",
        ['Dry', 'Damaged', 'Wet', 'Under Construction']
    )

    lighting = st.selectbox(
        "💡 Lighting Condition",
        ['Dark' ,'Daylight', 'Dawn' ,'Dusk']
    )

    traffic_control = st.selectbox(
        "🚦 Traffic Control",
        ['Signals', 'Police Checkpost', 'Signs']
    )

with col2:

    speed_limit = st.slider(
        "🚗 Speed Limit",
        20,
        150,
        60
    )

    driver_age = st.slider(
        "👨 Driver Age",
        18,
        80,
        30
    )

    alcohol = st.selectbox(
        "🍺 Alcohol Involvement",
        ['Yes','No']
    )

    vehicle = st.selectbox(
        "🚘 Vehicle Type",
        ['Bus' ,'Truck' ,'Car' ,'Cycle', 'Auto-Rickshaw' ,'Two-Wheeler', 'Pedestrian']
    )

    time_of_day = st.selectbox(
        "🌙 Time of Day",
        ['Morning', 'Afternoon', 'Evening', 'Night']
    )

# -----------------------------
# PREDICTION BUTTON
# -----------------------------

if st.button("🔍 Predict Accident Severity"):
    

    input_dict = {
    'Weather Conditions': weather,
    'Road Type': road_type,
    'Road Condition': road_condition,
    'Lighting Conditions': lighting,
    'Traffic Control Presence': traffic_control,
    'Speed Limit (km/h)': speed_limit,
    'Driver Age': driver_age,
    'Alcohol Involvement': alcohol,
    'Vehicle Type Involved': vehicle,
    'Time Category': time_of_day
}
    

    input_df = pd.DataFrame([input_dict])

# Encode categorical columns

    for column in input_df.columns:

        if column in label_encoders:

            input_df[column] = label_encoders[column].transform(input_df[column])

# Predict

    prediction = model.predict(input_df)

    result = target_encoder.inverse_transform(prediction)[0]



    # -----------------------------
    # RISK DISPLAY
    # -----------------------------

    if result.lower() in ['fatal', 'severe', 'serious']:

        st.markdown(
            f'''
            <div class="prediction-box high-risk">
            🚨 HIGH RISK ACCIDENT POSSIBILITY
            <br><br>
            Severity: {result}
            </div>
            ''',
            unsafe_allow_html=True
        )

        voice_message = (
            "Warning! High accident severity risk detected."
        )

    elif result.lower() == 'moderate':

        st.markdown(
            f'''
            <div class="prediction-box medium-risk">
            ⚠ MODERATE RISK DETECTED
            <br><br>
            Severity: {result}
            </div>
            ''',
            unsafe_allow_html=True
        )

        voice_message = (
            "Moderate accident risk detected."
        )

    else:

        st.markdown(
            f'''
            <div class="prediction-box low-risk">
            ✅ LOW RISK DETECTED
            <br><br>
            Severity: {result}
            </div>
            ''',
            unsafe_allow_html=True
        )

        voice_message = (
            "Low accident risk detected."
        )

    # -----------------------------
    # SAFETY RECOMMENDATIONS
    # -----------------------------

    st.subheader("🦺 Safety Recommendations")

    if result.lower() in ['fatal', 'severe', 'serious']:

        st.error("""
        - Reduce speed immediately
        - Avoid rash driving
        - Maintain safe distance
        - Avoid driving in poor visibility
        """)

    elif result.lower() == 'moderate':

        st.warning("""
        - Drive carefully
        - Follow traffic signals
        - Maintain moderate speed
        """)

    else:

        st.success("""
        - Continue safe driving
        - Follow traffic rules
        """)

    # -----------------------------
    # VOICE ALERT
    # -----------------------------

    engine = pyttsx3.init()
    engine.setProperty('rate', 150)

    engine.say(voice_message)
    engine.runAndWait()

# -----------------------------
# FOOTER
# -----------------------------

st.markdown("---")

st.markdown("""
<center>

Developed using:
Python • Random Forest • Streamlit • Machine Learning

</center>
""", unsafe_allow_html=True)