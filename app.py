from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
from datetime import datetime
import pytz
from config import Config

app = Flask(__name__)
app.config.from_object(Config)

# Global variables for model and scaler
model = None
scaler = None

def load_or_train_model():
    """Load existing model or train a new one with sample data"""
    global model, scaler
    
    try:
        # Try to load existing model
        model = joblib.load('breast_cancer_model.pkl')
        scaler = joblib.load('breast_cancer_scaler.pkl')
        print("Model loaded successfully")
    except FileNotFoundError:
        # Train a basic model with sample data if no saved model exists
        print("Training new model...")
        
        # Create sample data (in production, use real dataset)
        np.random.seed(42)
        n_samples = 500
        n_features = 10
        
        # Generate synthetic features
        X = np.random.randn(n_samples, n_features)
        # Create target variable (0: benign, 1: malignant)
        y = (X.sum(axis=1) + np.random.randn(n_samples) * 0.5 > 0).astype(int)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = LogisticRegression(random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model accuracy: {accuracy:.2f}")
        
        # Save model and scaler
        joblib.dump(model, 'breast_cancer_model.pkl')
        joblib.dump(scaler, 'breast_cancer_scaler.pkl')
        print("Model saved successfully")

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Make prediction based on input features"""
    try:
        # Get JSON data from request
        data = request.get_json()
        
        # Extract features (expecting 10 features for this example)
        features = []
        for i in range(10):
            feature_key = f'feature_{i+1}'
            if feature_key in data:
                features.append(float(data[feature_key]))
            else:
                return jsonify({'error': f'Missing {feature_key}'}), 400
        
        # Convert to numpy array and reshape
        features_array = np.array(features).reshape(1, -1)
        
        # Scale features
        features_scaled = scaler.transform(features_array)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0]
        
        # Get current timestamp
        timezone = pytz.timezone(app.config['TIMEZONE'])
        timestamp = datetime.now(timezone).strftime('%Y-%m-%d %H:%M:%S %Z')
        
        result = {
            'prediction': 'Malignant' if prediction == 1 else 'Benign',
            'probability': {
                'benign': float(probability[0]),
                'malignant': float(probability[1])
            },
            'timestamp': timestamp,
            'confidence': float(max(probability))
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    timezone = pytz.timezone(app.config['TIMEZONE'])
    current_time = datetime.now(timezone).strftime('%Y-%m-%d %H:%M:%S %Z')
    
    return jsonify({
        'status': 'healthy',
        'timestamp': current_time,
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None
    })

@app.route('/model-info')
def model_info():
    """Get model information"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 404
    
    return jsonify({
        'model_type': type(model).__name__,
        'features_count': model.n_features_in_,
        'classes': model.classes_.tolist()
    })

# Template for basic HTML interface
@app.template_global()
def create_index_template():
    """Create basic HTML template content"""
    return """
<!DOCTYPE html>
<html>
<head>
    <title>Breast Cancer Predictor</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .container { max-width: 600px; margin: 0 auto; }
        .form-group { margin: 15px 0; }
        label { display: block; margin-bottom: 5px; }
        input { width: 100%; padding: 8px; margin-bottom: 10px; }
        button { background: #007bff; color: white; padding: 10px 20px; border: none; cursor: pointer; }
        .result { margin-top: 20px; padding: 15px; background: #f8f9fa; border-radius: 5px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Breast Cancer Predictor</h1>
        <form id="predictionForm">
            {% for i in range(10) %}
            <div class="form-group">
                <label for="feature_{{ i+1 }}">Feature {{ i+1 }}:</label>
                <input type="number" step="any" id="feature_{{ i+1 }}" name="feature_{{ i+1 }}" required>
            </div>
            {% endfor %}
            <button type="submit">Predict</button>
        </form>
        <div id="result" class="result" style="display:none;"></div>
    </div>
    
    <script>
        document.getElementById('predictionForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const formData = new FormData(e.target);
            const data = {};
            for (let [key, value] of formData.entries()) {
                data[key] = parseFloat(value);
            }
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                });
                
                const result = await response.json();
                const resultDiv = document.getElementById('result');
                
                if (response.ok) {
                    resultDiv.innerHTML = `
                        <h3>Prediction: ${result.prediction}</h3>
                        <p>Confidence: ${(result.confidence * 100).toFixed(2)}%</p>
                        <p>Benign Probability: ${(result.probability.benign * 100).toFixed(2)}%</p>
                        <p>Malignant Probability: ${(result.probability.malignant * 100).toFixed(2)}%</p>
                        <p>Timestamp: ${result.timestamp}</p>
                    `;
                } else {
                    resultDiv.innerHTML = `<p style="color: red;">Error: ${result.error}</p>`;
                }
                
                resultDiv.style.display = 'block';
            } catch (error) {
                document.getElementById('result').innerHTML = `<p style="color: red;">Error: ${error.message}</p>`;
                document.getElementById('result').style.display = 'block';
            }
        });
    </script>
</body>
</html>
    """

# Create templates directory and file if they don't exist
import os
if not os.path.exists('templates'):
    os.makedirs('templates')

# Write the template file
template_content = """<!DOCTYPE html>
<html>
<head>
    <title>Breast Cancer Predictor</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .container { max-width: 600px; margin: 0 auto; }
        .form-group { margin: 15px 0; }
        label { display: block; margin-bottom: 5px; }
        input { width: 100%; padding: 8px; margin-bottom: 10px; }
        button { background: #007bff; color: white; padding: 10px 20px; border: none; cursor: pointer; }
        .result { margin-top: 20px; padding: 15px; background: #f8f9fa; border-radius: 5px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Breast Cancer Predictor</h1>
        <form id="predictionForm">
            {% for i in range(10) %}
            <div class="form-group">
                <label for="feature_{{ i+1 }}">Feature {{ i+1 }}:</label>
                <input type="number" step="any" id="feature_{{ i+1 }}" name="feature_{{ i+1 }}" required>
            </div>
            {% endfor %}
            <button type="submit">Predict</button>
        </form>
        <div id="result" class="result" style="display:none;"></div>
    </div>
    
    <script>
        document.getElementById('predictionForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const formData = new FormData(e.target);
            const data = {};
            for (let [key, value] of formData.entries()) {
                data[key] = parseFloat(value);
            }
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                });
                
                const result = await response.json();
                const resultDiv = document.getElementById('result');
                
                if (response.ok) {
                    resultDiv.innerHTML = `
                        <h3>Prediction: ${result.prediction}</h3>
                        <p>Confidence: ${(result.confidence * 100).toFixed(2)}%</p>
                        <p>Benign Probability: ${(result.probability.benign * 100).toFixed(2)}%</p>
                        <p>Malignant Probability: ${(result.probability.malignant * 100).toFixed(2)}%</p>
                        <p>Timestamp: ${result.timestamp}</p>
                    `;
                } else {
                    resultDiv.innerHTML = `<p style="color: red;">Error: ${result.error}</p>`;
                }
                
                resultDiv.style.display = 'block';
            } catch (error) {
                document.getElementById('result').innerHTML = `<p style="color: red;">Error: ${error.message}</p>`;
                document.getElementById('result').style.display = 'block';
            }
        });
    </script>
</body>
</html>"""

with open('templates/index.html', 'w') as f:
    f.write(template_content)

if __name__ == '__main__':
    # Load or train model on startup
    load_or_train_model()
    
    # Run the Flask app
    app.run(
        host=app.config['HOST'],
        port=app.config['PORT'],
        debug=app.config['DEBUG']
    )