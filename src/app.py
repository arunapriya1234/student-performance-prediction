
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# Locate the main project folder
BASE_DIR = Path(__file__).resolve().parent.parent

# Configure the webpage
st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="centered"
)


# Load the saved model and features
@st.cache_resource
def load_model():
    model_path = BASE_DIR / "student_performance_model.pkl"
    features_path = BASE_DIR / "model_features.pkl"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}. "
            "Please run src/save_model.py first."
        )

    if not features_path.exists():
        raise FileNotFoundError(
            f"Feature file not found: {features_path}. "
            "Please run src/save_model.py first."
        )

    model = joblib.load(model_path)
    features = joblib.load(features_path)

    return model, features


# Load the dataset
@st.cache_data
def load_dataset():
    data_path = BASE_DIR / "data" / "student-mat.csv"

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    return pd.read_csv(data_path, sep=";")


# Page heading
st.title("🎓 Student Performance Prediction")

st.write(
    "Explore student information and estimate the final grade "
    "using a trained machine learning model."
)

st.info(
    "This model predicts G3 using student characteristics and "
    "study-related information. Earlier grades G1 and G2 are excluded."
)

try:
    model, features = load_model()
    df = load_dataset()

    # Select a sample student
    st.subheader("Select a Student")

    student_index = st.number_input(
        "Student record number",
        min_value=0,
        max_value=len(df) - 1,
        value=0,
        step=1
    )

    # Remove earlier grades and target grade
    student = df.iloc[[int(student_index)]].drop(
        columns=["G1", "G2", "G3"]
    )

    # Match the exact columns used during training
    student = student[features]

    st.subheader("Student Details")

    st.dataframe(
        student.T.rename(columns={student.index[0]: "Value"}),
        use_container_width=True
    )

    # Predict final grade
    if st.button("Predict Final Grade", type="primary"):
        prediction = float(model.predict(student)[0])

        # Keep displayed grade within the valid range
        prediction = max(0.0, min(20.0, prediction))

        st.subheader("Prediction Result")

        st.metric(
            label="Estimated Final Grade",
            value=f"{prediction:.2f} / 20"
        )

        st.progress(prediction / 20)

        if prediction >= 10:
            st.success("The predicted grade is 10 or above.")
        else:
            st.warning("The predicted grade is below 10.")

        st.caption(
            "This is a model estimate, not a guaranteed grade "
            "or an official academic assessment."
        )

except FileNotFoundError as error:
    st.error(str(error))

except Exception as error:
    st.error(f"Something went wrong: {error}")
