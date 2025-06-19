from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from datetime import datetime
import numpy as np
import traceback
import logging
import os

# Import your custom predictor class
from ml.predict import BreastCancerPredictor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configuration
app.config['JSON_SORT_KEYS'] = False
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Load the model
predictor = None
try:
    predictor = BreastCancerPredictor(model_path="ml/model.pkl")
    models_loaded = True
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    models_loaded = False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy' if models_loaded else 'model_not_loaded',
        'models_loaded': models_loaded,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/predict', methods=['POST'])
def predict():
    if not models_loaded or predictor is None:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        data = request.get_json() or request.form.to_dict()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        # Detect key format
        possible_formats = ['feature_{}', 'feature{}', '{}', 'f{}']
        working_format = None
        for fmt in possible_formats:
            if fmt.format(1) in data:
                working_format = fmt
                break

        feature_mapping = {}
        if not working_format:
            keys = sorted([k for k in data.keys() if k.replace('_', '').isdigit()])
            if len(keys) >= 30:
                for i, k in enumerate(keys[:30]):
                    feature_mapping[i + 1] = k
            else:
                return jsonify({'error': 'Invalid key format', 'expected': possible_formats}), 400

        features = []
        for i in range(1, 31):
            key = working_format.format(i) if working_format else feature_mapping[i]
            val = data.get(key)
            try:
                val = float(val)
                if not np.isfinite(val):
                    raise ValueError
                features.append(val)
            except:
                return jsonify({'error': f'Invalid or missing value for feature {i}', 'key': key}), 400

        result = predictor.predict(features)

        return jsonify({
            'success': True,
            'prediction': result['prediction'],
            'confidence': result['confidence'],
            'probability': result['probability'],
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'model_version': '1.0'
            }
        })

    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({'error': 'Server error', 'details': str(e)}), 500

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large'}), 413

@app.errorhandler(400)
def bad_request(e):
    return jsonify({'error': 'Bad request', 'message': str(e)}), 400

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    host = os.environ.get('HOST', '0.0.0.0')
    debug = os.environ.get('DEBUG', 'True').lower() == 'true'

    logger.info(f"Starting Flask app on {host}:{port} (debug={debug})")
    app.run(host=host, port=port, debug=debug)
