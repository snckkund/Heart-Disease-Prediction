from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class ModelTrainer:
    def __init__(self, processor):
        """Initialize the model trainer with a data processor"""
        self.processor = processor
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )

    def train_and_evaluate(self):
        """Train the model and evaluate its performance"""
        # Train model
        self.model.fit(self.processor.X_train, self.processor.y_train)

        # Make predictions
        y_pred = self.model.predict(self.processor.X_test)

        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(self.processor.y_test, y_pred),
            'precision': precision_score(self.processor.y_test, y_pred),
            'recall': recall_score(self.processor.y_test, y_pred),
            'f1': f1_score(self.processor.y_test, y_pred)
        }

        return metrics

    def predict_with_proba(self, X):
        """Make prediction with probability estimates"""
        try:
            prediction = self.model.predict(X)[0]
            probabilities = self.model.predict_proba(X)[0]
            return prediction, probabilities[1]  # Return prediction and probability of class 1
        except Exception as e:
            raise ValueError(f"Error in prediction: {str(e)}")
