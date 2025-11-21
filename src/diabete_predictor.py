import joblib
import pandas as pd
import numpy as np


class DiabetesPredictor:
    """Lightweight class for fast predictions - perfect for web apps"""

    def __init__(self, model_path='../resource/model/diabetes_model.pkl'):
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.performance = None

        self.load_model()

    def load_model(self):
        """Load pre-trained model (FAST - no training)"""
        try:
            model_data = joblib.load(self.model_path)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
            self.performance = model_data.get('performance', {})
            print("✅ Model loaded successfully!")
            print(f"   Accuracy: {self.performance.get('accuracy', 'N/A')}")
            print(f"   ROC-AUC: {self.performance.get('roc_auc', 'N/A')}")
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file not found: {self.model_path}. Please run model_trainer.py first.")

    def predict(self, patient_data):
        """Fast prediction for web app"""
        if self.model is None:
            raise ValueError("Model not loaded!")

        # Convert to DataFrame with correct feature order
        patient_df = pd.DataFrame([patient_data], columns=self.feature_names)
        patient_df = patient_df[self.feature_names]  # Ensure correct order

        # Scale features
        patient_scaled = self.scaler.transform(patient_df)

        # Predict
        prediction = self.model.predict(patient_scaled)[0]
        probability = self.model.predict_proba(patient_scaled)[0][1]

        return {
            'prediction': int(prediction),
            'probability': float(probability),
            'risk_level': self._get_risk_level(probability),
            'message': self._get_recommendation(probability)
        }

    def _get_risk_level(self, probability):
        if probability >= 0.7:
            return "HIGH"
        elif probability >= 0.4:
            return "MEDIUM"
        else:
            return "LOW"

    def _get_recommendation(self, probability):
        if probability >= 0.7:
            return "Cần gặp bác sĩ chuyên khoa ngay lập tức"
        elif probability >= 0.4:
            return "Nên kiểm tra định kỳ và điều chỉnh lối sống"
        else:
            return "Duy trì lối sống lành mạnh"


# Singleton instance for web app
predictor_instance = None


def get_predictor():
    """Get singleton predictor instance (for web app)"""
    global predictor_instance
    if predictor_instance is None:
        predictor_instance = DiabetesPredictor()
    return predictor_instance


# Demo usage for web app
if __name__ == "__main__":
    # This runs instantly - no training!
    predictor = get_predictor()

    # Test prediction
    test_patient = {
        'Pregnancies': 2,
        'Glucose': 150,
        'BloodPressure': 85,
        'SkinThickness': 30,
        'Insulin': 180,
        'BMI': 32.0,
        'DiabetesPedigreeFunction': 0.7,
        'Age': 45
    }

    result = predictor.predict(test_patient)
    print("\n🔍 Prediction Result:")
    print(f"Diagnosis: {'DIABETES' if result['prediction'] == 1 else 'NO DIABETES'}")
    print(f"Probability: {result['probability']:.1%}")
    print(f"Risk Level: {result['risk_level']}")
    print(f"Recommendation: {result['message']}")