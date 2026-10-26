
import joblib
import pandas as pd

# Load the saved model and expected input columns
model = joblib.load("student_performance_model.pkl")
features = joblib.load("model_features.pkl")

# Example student details
student = {
    "school": "GP",
    "sex": "F",
    "age": 17,
    "address": "U",
    "famsize": "GT3",
    "Pstatus": "T",
    "Medu": 3,
    "Fedu": 3,
    "Mjob": "services",
    "Fjob": "other",
    "reason": "course",
    "guardian": "mother",
    "traveltime": 1,
    "studytime": 3,
    "failures": 0,
    "schoolsup": "no",
    "famsup": "yes",
    "paid": "no",
    "activities": "yes",
    "nursery": "yes",
    "higher": "yes",
    "internet": "yes",
    "romantic": "no",
    "famrel": 4,
    "freetime": 3,
    "goout": 2,
    "Dalc": 1,
    "Walc": 2,
    "health": 4,
    "absences": 2
}

# Ensure columns are in the exact training order
input_df = pd.DataFrame([student])[features]

# Generate prediction
prediction = model.predict(input_df)[0]

# Grades range from 0 to 20
prediction = max(0, min(20, prediction))

print("\nStudent Performance Prediction")
print("--------------------------------")
print(f"Predicted final grade: {prediction:.2f}/20")
print("--------------------------------")
print("Note: This is an estimate, not a guaranteed result.")
