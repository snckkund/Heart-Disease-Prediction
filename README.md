# Heart Disease Predictor

## Description
The Heart Disease Predictor is a machine learning-based web application that predicts the likelihood of heart disease in patients based on various medical attributes. Built with Streamlit and scikit-learn, this tool provides healthcare professionals with quick and accurate predictions to assist in patient diagnosis.

## Project Structure
```
HeartDiseasePredictor/
│
├── app.py                 # Main Streamlit application file
├── data_processor.py      # Data preprocessing and handling
├── model_trainer.py       # Machine learning model implementation
├── visualization.py       # Data visualization functions
├── requirements.txt       # Project dependencies
│
├── data/                  # Dataset directory
│   └── heart_disease_data.csv
│
├── .streamlit/           # Streamlit configuration
│
└── README.md             # Project documentation
```

## Features
- Interactive web interface for data input and prediction
- Real-time prediction of heart disease probability
- Data visualization and analysis capabilities
- Pre-trained machine learning model
- Comprehensive data preprocessing pipeline

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/snckkund/Heart-Disease-Prediction.git
   cd Heart-Disease-Prediction
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Start the Streamlit application:
   ```bash
   streamlit run app.py
   ```

2. Open your web browser and navigate to the displayed URL (http://localhost:5000)

3. Input patient data in the web interface:
   - Age
   - Sex
   - Chest Pain Type
   - Blood Pressure
   - Cholesterol Levels
   - And other required medical attributes

4. Click "Predict" to get the heart disease prediction result

## Technical Details
- **Frontend**: Streamlit
- **Backend**: Python
- **Machine Learning**: scikit-learn
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn

## Model Information
The predictor uses a machine learning model trained on medical data with the following features:
- Demographic information
- Medical test results
- Patient symptoms
- Historical data