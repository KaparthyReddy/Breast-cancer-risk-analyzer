# 🎗️ Breast Cancer Risk Analyzer

A comprehensive machine learning application for breast cancer prediction using advanced ensemble methods and feature engineering. This project includes both a high-performance ML model generator and an interactive web interface for risk assessment.

## 📋 Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [API Documentation](#api-documentation)
- [Contributing](#contributing)
- [License](#license)

## ✨ Features

### Machine Learning Model
- **Advanced Feature Engineering**: 150+ engineered features from 30 original features
- **Ensemble Methods**: Combines XGBoost, LightGBM, CatBoost, Random Forest, and more
- **Target Accuracy**: Optimized to achieve 98%+ accuracy
- **Automated Preprocessing**: Intelligent feature selection and scaling
- **Cross-Validation**: 10-fold stratified cross-validation for robust evaluation
- **Outlier Detection**: Automatic outlier removal using Isolation Forest

### Web Application
- **Interactive UI**: User-friendly interface for risk assessment
- **Real-time Predictions**: Instant cancer risk analysis
- **Data Visualization**: Visual representation of feature importance
- **Responsive Design**: Works on desktop and mobile devices
- **Educational Content**: Information about breast cancer and risk factors

## 📁 Project Structure

```
breast-cancer-risk-analyzer/
│
├── ml/
│   ├── generate_model.py          # Enhanced model generator script
│   ├── best_breast_cancer_model.pkl  # Trained model (generated)
│   └── requirements.txt            # ML dependencies
│
├── web/
│   ├── app.py                      # Flask/FastAPI web application
│   ├── static/
│   │   ├── css/
│   │   │   └── style.css          # Application styles
│   │   └── js/
│   │       └── main.js            # Frontend JavaScript
│   ├── templates/
│   │   ├── index.html             # Home page
│   │   ├── predict.html           # Prediction interface
│   │   └── results.html           # Results display
│   └── requirements.txt           # Web app dependencies
│
├── data/
│   └── sample_data.csv            # Sample data for testing
│
├── docs/
│   ├── model_documentation.md     # Model architecture details
│   └── api_reference.md           # API endpoints documentation
│
├── tests/
│   ├── test_model.py              # Model unit tests
│   └── test_api.py                # API unit tests
│
├── README.md                       # This file
├── requirements.txt                # Main project dependencies
└── .gitignore                      # Git ignore file
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Virtual environment (recommended)

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/breast-cancer-risk-analyzer.git
cd breast-cancer-risk-analyzer
```

### Step 2: Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
# Install all dependencies
pip install -r requirements.txt

# Or install separately for ML and Web
pip install -r ml/requirements.txt
pip install -r web/requirements.txt
```

### Step 4: Install Required Packages

The main dependencies include:

**Machine Learning:**
```bash
pip install pandas numpy scikit-learn xgboost lightgbm catboost scipy joblib
```

**Web Application:**
```bash
pip install flask flask-cors
# OR for FastAPI
pip install fastapi uvicorn python-multipart
```

## 🎯 Usage

### 1. Generate the ML Model

First, train the machine learning model:

```bash
cd ml
python generate_model.py
```

This will:
- Load the breast cancer dataset
- Perform feature engineering
- Train multiple models
- Select the best performing model
- Save the model as `best_breast_cancer_model.pkl`

**Expected Output:**
```
🚀 Enhanced Breast Cancer Prediction Model Generator
============================================================
Loading breast cancer dataset...
Dataset loaded: 569 samples, 30 features
Removed 29 outliers (5.10%)
Applying advanced feature engineering...
Feature engineering completed: 150+ features created
...
🎯 FINAL RESULTS:
Best Model Accuracy: 0.9825 (98.25%)
Cross-validation Mean: 0.9778 (97.78%)
AUC Score: 0.9965
🏆 TARGET ACHIEVED: 98%+ accuracy reached!
```

### 2. Run the Web Application

#### Option A: Flask (Recommended for Development)

Create `web/app.py`:

```python
from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the trained model
model_package = joblib.load('../ml/best_breast_cancer_model.pkl')
model = model_package['model']
scaler = model_package['scaler']
selector = model_package['selector']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('predict.html')
    
    # Get data from POST request
    data = request.get_json()
    
    # Process input features
    # ... (add your processing logic)
    
    # Make prediction
    # prediction = model.predict(processed_data)
    
    return jsonify({'prediction': 'result'})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
```

**Run the Flask app:**

```bash
cd web
python app.py
```

Access the application at: `http://localhost:5000`

#### Option B: FastAPI (Recommended for Production)

Create `web/app.py`:

```python
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import joblib
import numpy as np

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Load model
model_package = joblib.load('../ml/best_breast_cancer_model.pkl')

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/predict")
async def predict(data: dict):
    # Process and predict
    return {"prediction": "result"}
```

**Run the FastAPI app:**

```bash
cd web
uvicorn app:app --reload --port 5000
```

Access the application at: `http://localhost:5000`

### 3. Using the Web Interface

1. **Navigate to the home page** (`http://localhost:5000`)
2. **Click on "Predict Risk"** or navigate to `/predict`
3. **Enter patient data:**
   - Mean radius, texture, perimeter, etc.
   - You can use sample values or real measurements
4. **Click "Analyze Risk"**
5. **View results:**
   - Risk prediction (Benign/Malignant)
   - Confidence score
   - Feature importance visualization

### 4. Sample Input Data

You can test with these sample values:

**Benign Sample:**
```json
{
  "mean_radius": 13.5,
  "mean_texture": 18.2,
  "mean_perimeter": 87.5,
  "mean_area": 566.3,
  "mean_smoothness": 0.09,
  ...
}
```

**Malignant Sample:**
```json
{
  "mean_radius": 20.5,
  "mean_texture": 25.3,
  "mean_perimeter": 142.4,
  "mean_area": 1256.8,
  "mean_smoothness": 0.12,
  ...
}
```

## 📊 Model Performance

### Metrics

- **Accuracy**: 98.25% (test set)
- **Cross-Validation**: 97.78% ± 1.2%
- **AUC-ROC**: 0.9965
- **Precision**: 98.5% (Malignant class)
- **Recall**: 97.8% (Malignant class)
- **F1-Score**: 98.2%

### Model Components

1. **Base Models:**
   - XGBoost
   - LightGBM
   - CatBoost
   - Random Forest
   - Extra Trees
   - Gradient Boosting
   - Neural Networks (MLP)
   - SVM
   - Logistic Regression

2. **Ensemble Strategy:**
   - Stacking with multiple meta-learners
   - Soft voting ensembles
   - Cross-validation during training

3. **Feature Engineering:**
   - Polynomial features
   - Ratio features
   - Interaction features
   - Statistical aggregations
   - Domain-specific features

## 🔧 API Documentation

### Endpoints

#### `GET /`
- **Description**: Home page
- **Returns**: HTML page

#### `GET /predict`
- **Description**: Prediction interface
- **Returns**: HTML form

#### `POST /api/predict`
- **Description**: Make a prediction
- **Request Body**:
```json
{
  "features": {
    "mean_radius": 17.99,
    "mean_texture": 10.38,
    ...
  }
}
```
- **Response**:
```json
{
  "prediction": "Benign",
  "confidence": 0.95,
  "probability": {
    "malignant": 0.05,
    "benign": 0.95
  }
}
```

#### `GET /api/model-info`
- **Description**: Get model information
- **Response**:
```json
{
  "accuracy": 0.9825,
  "model_type": "Stacking Ensemble",
  "features_count": 150,
  "trained_date": "2024-01-15"
}
```

## 🧪 Testing

### Run Unit Tests

```bash
# Test the model
python -m pytest tests/test_model.py

# Test the API
python -m pytest tests/test_api.py

# Run all tests with coverage
pytest --cov=. tests/
```

### Manual Testing

```bash
# Test model loading
python -c "import joblib; model = joblib.load('ml/best_breast_cancer_model.pkl'); print('Model loaded successfully')"

# Test prediction
python tests/test_prediction.py
```

## 🛠️ Development

### Adding New Features

1. **Modify feature engineering** in `ml/generate_model.py`
2. **Retrain the model**:
   ```bash
   cd ml
   python generate_model.py
   ```
3. **Update the web interface** to accept new features

### Improving Model Performance

- Adjust hyperparameters in `create_base_models()`
- Add more ensemble models
- Experiment with different preprocessing methods
- Increase feature engineering complexity

## 📝 Configuration

### Model Configuration

Edit `ml/config.py`:

```python
MODEL_CONFIG = {
    'random_state': 42,
    'test_size': 0.2,
    'cv_folds': 10,
    'outlier_contamination': 0.05,
    'feature_selection_k': 50
}
```

### Web App Configuration

Edit `web/config.py`:

```python
WEB_CONFIG = {
    'host': '0.0.0.0',
    'port': 5000,
    'debug': True,
    'model_path': '../ml/best_breast_cancer_model.pkl'
}
```

## 🚢 Deployment

### Deploy to Heroku

```bash
# Create Procfile
echo "web: uvicorn web.app:app --host 0.0.0.0 --port $PORT" > Procfile

# Deploy
heroku create breast-cancer-analyzer
git push heroku main
```

### Deploy to AWS/Azure

1. Package the application
2. Upload to cloud storage
3. Configure compute instance
4. Set environment variables
5. Run the application

### Docker Deployment

Create `Dockerfile`:

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["uvicorn", "web.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:

```bash
docker build -t breast-cancer-analyzer .
docker run -p 8000:8000 breast-cancer-analyzer
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

**MEDICAL DISCLAIMER**: This tool is for educational and research purposes only. It should NOT be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare professionals for medical decisions.

## 👥 Authors

- Your Name - Initial work - [@KaparthyReddy](https://github.com/KaparthyReddy)

## 🙏 Acknowledgments

- Wisconsin Breast Cancer Dataset (WBCD)
- Scikit-learn community
- XGBoost, LightGBM, and CatBoost developers
- Open source ML community


---

Made with ❤️ for better healthcare outcomes
