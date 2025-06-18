from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import os

app = Flask(__name__)

# Load the trained model and scaler
try:
    model = joblib.load('breast_cancer_model.pkl')
    scaler = joblib.load('breast_cancer_scaler.pkl')
    print("Model and scaler loaded successfully!")
except FileNotFoundError as e:
    print(f"Error loading model files: {e}")
    model = None
    scaler = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None:
        return jsonify({'error': 'Model not loaded properly'}), 500
    
    try:
        # Get JSON data from request
        data = request.get_json()
        
        # Add debugging
        print(f"Received data keys: {list(data.keys()) if data else 'No data'}")
        print(f"Data type: {type(data)}")
        print(f"Total fields received: {len(data) if data else 0}")
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Extract features in correct order (feature_1 to feature_30)
        features = []
        missing_features = []
        invalid_features = []
        
        for i in range(1, 31):
            feature_key = f'feature_{i}'
            if feature_key not in data:
                missing_features.append(feature_key)
            else:
                try:
                    value = data[feature_key]
                    # Handle different data types
                    if value is None or value == '':
                        missing_features.append(feature_key)
                    else:
                        float_value = float(value)
                        # Check for reasonable range (basic validation)
                        if float_value < 0:
                            invalid_features.append(f"{feature_key}: negative value ({float_value})")
                        else:
                            features.append(float_value)
                except (ValueError, TypeError) as e:
                    invalid_features.append(f"{feature_key}: invalid value ({data[feature_key]}) - {str(e)}")
        
        # Check for errors
        if missing_features:
            return jsonify({
                'error': f'Missing features: {missing_features}',
                'received_keys': list(data.keys()),
                'total_received': len(data)
            }), 400
        
        if invalid_features:
            return jsonify({
                'error': f'Invalid features: {invalid_features}'
            }), 400
        
        if len(features) != 30:
            return jsonify({
                'error': f'Expected 30 features, got {len(features)}',
                'missing_count': 30 - len(features),
                'received_keys': list(data.keys())
            }), 400
        
        print(f"Successfully extracted {len(features)} features")
        print(f"Feature sample: {features[:5]}...")  # Show first 5 features
        
        # Convert to numpy array and reshape
        features_array = np.array(features).reshape(1, -1)
        
        # Scale the features
        features_scaled = scaler.transform(features_array)
        
        # Make prediction
        prediction_proba = model.predict_proba(features_scaled)[0]
        prediction = model.predict(features_scaled)[0]
        
        # Map prediction to readable format
        prediction_label = 'Malignant' if prediction == 1 else 'Benign'
        confidence = max(prediction_proba)
        
        # Prepare response
        response = {
            'prediction': prediction_label,
            'confidence': float(confidence),
            'probability': {
                'benign': float(prediction_proba[0]),
                'malignant': float(prediction_proba[1])
            },
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'features_processed': len(features)
        }
        
        print(f"Prediction successful: {prediction_label} with {confidence:.3f} confidence")
        
        return jsonify(response)
        
    except ValueError as e:
        print(f"ValueError: {e}")
        return jsonify({'error': f'Invalid input data: {str(e)}'}), 400
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500

# Add a test endpoint to debug what's being received
@app.route('/debug', methods=['POST'])
def debug():
    data = request.get_json()
    return jsonify({
        'received_data': data,
        'data_type': str(type(data)),
        'keys': list(data.keys()) if data else None,
        'values_sample': {k: v for k, v in list(data.items())[:5]} if data else None,
        'total_fields': len(data) if data else 0
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)