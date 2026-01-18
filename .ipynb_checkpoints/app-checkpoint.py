import streamlit as st
import pandas as pd
import joblib

# Load the trained model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("🎓 Student Success Indicator App")
st.write("Fill in the details below to predict if a student will be successful (Final Grade >= 10).")

# Input form
input_data = {
    "school": st.selectbox("School (0: GP, 1: MS)", [0, 1]),
    "sex": st.selectbox("Sex (0: Female, 1: Male)", [0, 1]),
    "age": st.number_input("Age", 15, 22),
    "address": st.selectbox("Address (0: Rural, 1: Urban)", [0, 1]),
    "famsize": st.selectbox("Family Size (0: LE3, 1: GT3)", [0, 1]),
    "Pstatus": st.selectbox("Parent Status (0: A, 1: T)", [0, 1]),
    "Medu": st.slider("Mother's Education (0-4)", 0, 4),
    "Fedu": st.slider("Father's Education (0-4)", 0, 4),
    "Mjob": st.selectbox("Mother's Job (encoded)", [0, 1, 2, 3, 4]),
    "Fjob": st.selectbox("Father's Job (encoded)", [0, 1, 2, 3, 4]),
    "reason": st.selectbox("Reason for School Choice (encoded)", [0, 1, 2, 3]),
    "guardian": st.selectbox("Guardian (encoded)", [0, 1, 2]),
    "traveltime": st.slider("Travel Time (1-4)", 1, 4),
    "studytime": st.slider("Study Time (1-4)", 1, 4),
    "failures": st.slider("Past Class Failures (0-3)", 0, 3),
    "schoolsup": st.selectbox("Extra Educational Support (0: No, 1: Yes)", [0, 1]),
    "famsup": st.selectbox("Family Educational Support (0: No, 1: Yes)", [0, 1]),
    "paid": st.selectbox("Extra Paid Classes (0: No, 1: Yes)", [0, 1]),
    "activities": st.selectbox("Extracurricular Activities (0: No, 1: Yes)", [0, 1]),
    "nursery": st.selectbox("Attended Nursery (0: No, 1: Yes)", [0, 1]),
    "higher": st.selectbox("Wants Higher Education (0: No, 1: Yes)", [0, 1]),
    "internet": st.selectbox("Internet Access (0: No, 1: Yes)", [0, 1]),
    "romantic": st.selectbox("In a Romantic Relationship (0: No, 1: Yes)", [0, 1]),
    "famrel": st.slider("Family Relationship Quality (1-5)", 1, 5),
    "freetime": st.slider("Free Time (1-5)", 1, 5),
    "goout": st.slider("Going Out Frequency (1-5)", 1, 5),
    "Dalc": st.slider("Workday Alcohol Consumption (1-5)", 1, 5),
    "Walc": st.slider("Weekend Alcohol Consumption (1-5)", 1, 5),
    "health": st.slider("Health Status (1-5)", 1, 5),
    "absences": st.slider("Absences", 0, 100),
    "G1": st.slider("First Semester Grade (0-20)", 0, 20),
    "G2": st.slider("Second Semester Grade (0-20)", 0, 20)
}

# Convert to DataFrame
input_df = pd.DataFrame([input_data])

# Apply scaler
input_scaled = scaler.transform(input_df)

# Make prediction
prediction = model.predict(input_scaled)[0]
result = "✅ Successful" if prediction == 1 else "❌ Not Successful"

# Show result
st.subheader("Prediction:")
st.write(result)
