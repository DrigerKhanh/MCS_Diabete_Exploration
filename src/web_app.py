import os

from flask import Flask, render_template, request, jsonify
from diabete_predictor import get_predictor
import json

app = Flask(__name__,
           template_folder=os.path.join(os.path.dirname(__file__), '..', 'resource', 'template'))
predictor = get_predictor()


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from form
        patient_data = {
            'Pregnancies': int(request.form['pregnancies']),
            'Glucose': float(request.form['glucose']),
            'BloodPressure': float(request.form['blood_pressure']),
            'SkinThickness': float(request.form['skin_thickness']),
            'Insulin': float(request.form['insulin']),
            'BMI': float(request.form['bmi']),
            'DiabetesPedigreeFunction': float(request.form['pedigree']),
            'Age': int(request.form['age'])
        }

        # Make prediction
        result = predictor.predict(patient_data)

        return jsonify({
            'success': True,
            'diagnosis': 'CÓ TIỂU ĐƯỜNG' if result['prediction'] == 1 else 'KHÔNG TIỂU ĐƯỜNG',
            'probability': f"{result['probability']:.1%}",
            'risk_level': result['risk_level'],
            'recommendation': result['message'],
            'patient_data': patient_data
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for mobile apps"""
    try:
        data = request.get_json()
        result = predictor.predict(data)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)