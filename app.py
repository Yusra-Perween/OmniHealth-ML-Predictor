import streamlit as st
import numpy as np
import pandas as pd
import datetime
from joblib import load
import os
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Load ML model efficiently using Streamlit caching
@st.cache_resource
def load_ml_model():
    try:
        return load("models/health_model.pkl")
    except Exception as e:
        st.error(f"⚠️ Critical Error: Could not load the machine learning model. Details: {e}")
        return None

model = load_ml_model()

# App Title
st.set_page_config(page_title="Arogya AI", layout="centered")
st.title("🩺 SmartRx - Your Personal Health Assistant")

# Sidebar Input Function
def user_input_features():
    """
    Collects user health data via Streamlit sidebar inputs.
    
    Returns:
        pd.DataFrame: A single-row DataFrame containing the collected features.
    """
    st.sidebar.header("📝 Enter Health Details")
    pregnancies = st.sidebar.number_input("Pregnancies", 0, 20, 1)
    glucose = st.sidebar.slider("Glucose Level", 0, 200, 100)
    blood_pressure = st.sidebar.slider("Blood Pressure", 0, 122, 70)
    skin_thickness = st.sidebar.slider("Skin Thickness", 0, 100, 20)
    insulin = st.sidebar.slider("Insulin", 0, 846, 80)
    bmi = st.sidebar.slider("BMI", 0.0, 70.0, 25.0)
    dpf = st.sidebar.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
    age = st.sidebar.slider("Age", 10, 100, 30)

    data = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': blood_pressure,
        'SkinThickness': skin_thickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': dpf,
        'Age': age
    }
    return pd.DataFrame(data, index=[0])

# Save user input to history CSV
def save_history(data):
    """
    Saves the user input data to a local CSV file with a timestamp.
    
    Args:
        data (dict): A dictionary of health metrics submitted by the user.
    """
    data['Date'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df = pd.DataFrame([data])
    file_exists = os.path.isfile('data/health_history.csv')
    df.to_csv('data/health_history.csv', mode='a', header=not file_exists, index=False)

# Early Disease Detection
def detect_early_disease(features):
    """
    Analyzes health features for early signs of common medical risks.
    
    Args:
        features (pd.DataFrame): The user's input health features.
        
    Returns:
        str: A formatted string containing early warnings, or a success message.
    """
    warnings = []

    if features['Glucose'][0] > 140:
        warnings.append("🔴 High glucose level → Possible diabetes risk.")
    if features['BloodPressure'][0] > 90:
        warnings.append("⚠️ High blood pressure → Possible hypertension.")
    if features['BMI'][0] > 30:
        warnings.append("🧠 BMI above normal → Obesity risk.")
    if features['Age'][0] > 60:
        warnings.append("🧓 Age > 60 → Get regular checkups.")
    if features['Insulin'][0] > 200:
        warnings.append("🧪 High insulin level → Insulin resistance.")
    if features['SkinThickness'][0] > 35:
        warnings.append("📏 Skin thickness high → Possible insulin resistance.")
    if features['DiabetesPedigreeFunction'][0] > 1.0:
        warnings.append("🧬 Family history of diabetes is strong.")
    if features['Pregnancies'][0] > 2 and features['Glucose'][0] > 120:
        warnings.append("👶 Gestational diabetes risk (multiple pregnancies + high glucose).")

    return "✅ No immediate early disease risk detected." if not warnings else "\n".join(warnings)

# Smart Diet Suggestions
def suggest_diet(glucose, bmi):
    """
    Provides dietary suggestions based on glucose level and BMI.
    
    Args:
        glucose (float): The user's glucose level.
        bmi (float): The user's Body Mass Index.
        
    Returns:
        list: A list of actionable dietary recommendations.
    """
    if glucose > 140 or bmi > 30:
        return [
            "🥦 Eat more vegetables & fiber-rich foods",
            "🚫 Avoid sugary snacks & fried items",
            "💧 Drink more water (3+ litres)",
            "🏃‍♂️ Include daily 30 min walking/exercise"
        ]
    else:
        return [
            "🍎 Maintain a balanced diet with fruits, veggies",
            "🥗 Include lean proteins & whole grains",
            "🚶‍♀️ Stay active and hydrated"
        ]

# Tabs: UI Sections
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Prediction", "🧠 Early Detection", "🍽️ Diet", "📊 Dashboard", "📜 History"])

# Get input
input_df = user_input_features()

# 🔍 TAB 1: Diabetes Prediction
with tab1:
    st.subheader("🔍 Input Health Data")
    st.write(input_df)

    if model is not None:
        prediction = model.predict(input_df)[0]
    
        st.subheader("🧠 Diabetes Risk Prediction")
        if prediction == 1:
            st.error("⚠️ You may be at **risk of diabetes**. Please consult a doctor.")
        else:
            st.success("✅ You are likely **not at risk** of diabetes.")
    else:
        st.error("Prediction unavailable because the model failed to load.")

# ⚠️ TAB 2: Early Disease Detection
with tab2:
    st.subheader("⚠️ Early Disease Risk Report")
    result = detect_early_disease(input_df)
    if "No immediate" in result:
        st.success(result)
    else:
        st.warning(result)

# 🍽️ TAB 3: Smart Diet Suggestion
with tab3:
    st.subheader("🍽️ Smart Diet Suggestions")
    diet = suggest_diet(input_df['Glucose'][0], input_df['BMI'][0])
    for tip in diet:
        st.markdown(f"- {tip}")

# 📊 TAB 4: Health Insights Dashboard
with tab4:
    st.title("📊 Health Insights Dashboard")

    try:
        df = pd.read_csv("data/health_data.csv")

        st.markdown("### 📌 Dataset Overview")
        st.dataframe(df.head())

        # Histogram for each metric
        st.markdown("### 📊 Distribution Charts")
        metrics = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'Age']
        for metric in metrics:
            fig = px.histogram(df, x=metric, nbins=30, title=f"{metric} Distribution", color_discrete_sequence=['indianred'])
            st.plotly_chart(fig)

        # Correlation heatmap
        st.markdown("### 🔗 Correlation Heatmap")
        corr = df.corr()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

        # Glucose trend across age group
        st.markdown("### 📈 Avg Glucose by Age Group")
        df['AgeGroup'] = pd.cut(df['Age'], bins=[20, 30, 40, 50, 60, 100], labels=["21-30", "31-40", "41-50", "51-60", "61+"])
        avg_glucose = df.groupby('AgeGroup')['Glucose'].mean().reset_index()
        fig = px.line(avg_glucose, x='AgeGroup', y='Glucose', markers=True, title="Avg Glucose vs Age Group")
        st.plotly_chart(fig)

        # Scatter plots
        st.markdown("### 🔁 Scatter Plots")
        col1, col2 = st.columns(2)
        with col1:
            fig = px.scatter(df, x='Glucose', y='Insulin', color='Outcome', title="Glucose vs Insulin")
            st.plotly_chart(fig)
        with col2:
            fig = px.scatter(df, x='BMI', y='Age', color='Outcome', title="BMI vs Age")
            st.plotly_chart(fig)

        # Diabetes outcome by age group
        st.markdown("### 🧑‍⚕️ Diabetes Count by Age Group")
        outcome_count = df.groupby(['AgeGroup', 'Outcome']).size().reset_index(name='Count')
        fig = px.bar(outcome_count, x='AgeGroup', y='Count', color='Outcome', barmode='group', title="Diabetes Outcome by Age Group")
        st.plotly_chart(fig)

    except Exception as e:
        st.error(f"⚠️ Unable to load data: {e}")

# 📜 TAB 5: History
with tab5:
    st.subheader("📜 Your Last 10 Records")
    try:
        save_history(input_df.iloc[0].to_dict())
        history_df = pd.read_csv("data/health_history.csv")
        st.dataframe(history_df.tail(10))
    except Exception as e:
        st.warning("No history found or error reading file.")
