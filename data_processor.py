import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class DataProcessor:
    def __init__(self, data):
        """Initialize the DataProcessor with input data"""
        self.data = self._validate_data(data)
        self.scaler = StandardScaler()
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.process_data()

    def _validate_data(self, data):
        """Validate and clean the input data"""
        try:
            # Remove any rows with missing values
            data = data.dropna()

            # Ensure all columns are numeric
            for col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')

            # Drop any rows with NaN after conversion
            data = data.dropna()

            # Store feature names
            self.feature_names = data.columns.tolist()

            return data
        except Exception as e:
            raise ValueError(f"Error in data validation: {str(e)}")

    def process_data(self):
        """Process the raw data and prepare it for modeling"""
        try:
            # Separate features and target
            self.X = self.data.drop('target', axis=1)
            self.y = self.data['target']

            # Store feature names before scaling
            self.feature_names = self.X.columns.tolist()

            # Scale features
            self.X = pd.DataFrame(
                self.scaler.fit_transform(self.X),
                columns=self.feature_names
            )

            # Split data
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y, test_size=0.2, random_state=42
            )

            print(f"Data processed successfully. Features: {self.feature_names}")
        except Exception as e:
            raise ValueError(f"Error in data processing: {str(e)}")

    def transform_data(self, data):
        """Transform new data using the fitted scaler"""
        try:
            # Ensure input data has the same columns as training data
            expected_columns = self.feature_names
            if not all(col in data.columns for col in expected_columns):
                raise ValueError(f"Input data missing required columns. Expected: {expected_columns}")

            # Reorder columns to match training data
            data = data[expected_columns]

            # Transform the data
            transformed_data = pd.DataFrame(
                self.scaler.transform(data),
                columns=expected_columns
            )

            return transformed_data
        except Exception as e:
            raise ValueError(f"Error in data transformation: {str(e)}")