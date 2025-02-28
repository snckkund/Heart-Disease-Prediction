import streamlit as st
import pandas as pd
import numpy as np
from data_processor import DataProcessor
from model_trainer import ModelTrainer
from visualization import Visualizer

st.set_page_config(
    page_title="Heart Disease Prediction System",
    page_icon="❤️",
    layout="wide"
)

# Initialize session state
if 'model_trainer' not in st.session_state:
    st.session_state.model_trainer = None
if 'processor' not in st.session_state:
    st.session_state.processor = None

def main():
    st.title("❤️ Heart Disease Prediction System")

    # Sidebar
    st.sidebar.header("Navigation")
    page = st.sidebar.radio(
        "Select a Page",
        ["Data Analysis", "Model Training", "Prediction"]
    )

    # Load and process data
    try:
        data = pd.read_csv('data/heart_disease.csv')
        if 'processor' not in st.session_state or st.session_state.processor is None:
            processor = DataProcessor(data)
            st.session_state.processor = processor
        visualizer = Visualizer(data)

        if page == "Data Analysis":
            st.header("Exploratory Data Analysis")

            # Display basic statistics
            st.subheader("Dataset Overview")
            st.write(data.describe())

            # Correlation analysis
            st.subheader("Feature Correlations")
            visualizer.plot_correlation_heatmap()

            # Feature distributions
            st.subheader("Feature Distributions")
            visualizer.plot_feature_distributions()

            # Target distribution
            st.subheader("Target Distribution")
            visualizer.plot_target_distribution()

        elif page == "Model Training":
            st.header("Model Training and Evaluation")

            if st.button("Train Model"):
                with st.spinner("Training model..."):
                    trainer = ModelTrainer(st.session_state.processor)
                    metrics = trainer.train_and_evaluate()
                    st.session_state.model_trainer = trainer

                st.success("Model trained successfully!")

                # Display metrics
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Accuracy", f"{metrics['accuracy']:.2f}")
                col2.metric("Precision", f"{metrics['precision']:.2f}")
                col3.metric("Recall", f"{metrics['recall']:.2f}")
                col4.metric("F1 Score", f"{metrics['f1']:.2f}")

                # Feature importance
                st.subheader("Feature Importance")
                visualizer.plot_feature_importance(
                    trainer.model,
                    st.session_state.processor.feature_names
                )

        else:  # Prediction page
            st.header("Heart Disease Prediction")

            if st.session_state.model_trainer is None:
                st.warning("Please train the model first!")
                return

            # Input form
            with st.form("prediction_form"):
                col1, col2 = st.columns(2)

                with col1:
                    age = st.number_input("Age", min_value=20, max_value=100, value=50)
                    sex = st.selectbox("Sex", ["Female", "Male"])
                    cp = st.selectbox("Chest Pain Type", 
                                    ["Typical Angina", "Atypical Angina", 
                                     "Non-anginal Pain", "Asymptomatic"])
                    trestbps = st.number_input("Resting Blood Pressure (mmHg)", 
                                             min_value=90, max_value=200, value=120)
                    chol = st.number_input("Serum Cholesterol (mg/dl)", 
                                         min_value=100, max_value=600, value=200)
                    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])

                with col2:
                    restecg = st.selectbox("Resting ECG Results", 
                                         ["Normal", "ST-T Wave Abnormality", 
                                          "Left Ventricular Hypertrophy"])
                    thalach = st.number_input("Maximum Heart Rate", 
                                            min_value=60, max_value=220, value=150)
                    exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
                    oldpeak = st.number_input("ST Depression", 
                                            min_value=0.0, max_value=6.0, value=0.0)
                    slope = st.selectbox("ST Segment Slope", 
                                       ["Upward", "Flat", "Downward"])

                submit = st.form_submit_button("Predict")

            if submit:
                try:
                    # Create input data dictionary with exact column names
                    input_data = pd.DataFrame([{
                        'age': age,
                        'sex': 1 if sex == "Male" else 0,
                        'cp': {"Typical Angina": 0, 
                              "Atypical Angina": 1,
                              "Non-anginal Pain": 2,
                              "Asymptomatic": 3}[cp],
                        'trestbps': trestbps,
                        'chol': chol,
                        'fbs': 1 if fbs == "Yes" else 0,
                        'restecg': {"Normal": 0,
                                   "ST-T Wave Abnormality": 1,
                                   "Left Ventricular Hypertrophy": 2}[restecg],
                        'thalach': thalach,
                        'exang': 1 if exang == "Yes" else 0,
                        'oldpeak': oldpeak,
                        'slope': {"Upward": 0, "Flat": 1, "Downward": 2}[slope]
                    }])

                    # Process input data
                    processed_input = st.session_state.processor.transform_data(input_data)

                    # Make prediction using the new method
                    prediction, probability = st.session_state.model_trainer.predict_with_proba(processed_input)

                    # Display results
                    result_container = st.container()
                    with result_container:
                        if prediction == 1:
                            st.error("⚠️ Heart Disease Detected")
                            st.write(f"Probability of heart disease: {probability:.2%}")
                        else:
                            st.success("✅ No Heart Disease Detected")
                            st.write(f"Probability of heart disease: {probability:.2%}")

                        # Display input summary
                        st.subheader("Input Summary")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("Patient Demographics:")
                            st.write(f"- Age: {age}")
                            st.write(f"- Sex: {sex}")
                            st.write(f"- Chest Pain Type: {cp}")
                        with col2:
                            st.write("Key Measurements:")
                            st.write(f"- Blood Pressure: {trestbps} mmHg")
                            st.write(f"- Cholesterol: {chol} mg/dl")
                            st.write(f"- Max Heart Rate: {thalach}")

                except Exception as e:
                    st.error(f"Error making prediction: {str(e)}")
                    st.write("Debug information:")
                    st.write(f"Input data shape: {input_data.shape}")
                    st.write(f"Input columns: {input_data.columns.tolist()}")
                    st.write(f"Expected features: {st.session_state.processor.feature_names}")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()