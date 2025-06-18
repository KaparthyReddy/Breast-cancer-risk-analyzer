#!/usr/bin/env python3
"""
Script to generate the model.pkl file for the breast cancer predictor.
Run this script to create the trained model file.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

def create_model():
    """
    Create and train the breast cancer prediction model.
    """
    print("Creating breast cancer prediction model...")
    
    # Load the breast cancer dataset
    data = load_breast_cancer()
    X = data.data
    y = data.target
    feature_names = data.feature_names
    
    print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train the model
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2
    )
    
    print("Training model...")
    model.fit(X_train_scaled, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Model accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Malignant', 'Benign']))
    
    # Prepare model data for saving
    model_data = {
        'model': model,
        'scaler': scaler,
        'feature_names': feature_names.tolist(),
        'accuracy': accuracy,
        'target_names': ['Malignant', 'Benign']
    }
    
    # Create ml directory if it doesn't exist
    os.makedirs('ml', exist_ok=True)
    
    # Save the model
    model_path = os.path.join('ml', 'model.pkl')
    joblib.dump(model_data, model_path)
    
    print(f"\nModel saved to: {model_path}")
    print(f"Model file size: {os.path.getsize(model_path) / 1024:.2f} KB")
    
    return model_path

def verify_model(model_path):
    """
    Verify that the saved model can be loaded and works correctly.
    """
    print("\nVerifying saved model...")
    
    try:
        # Load the model
        model_data = joblib.load(model_path)
        model = model_data['model']
        scaler = model_data['scaler']
        
        # Test with sample data
        sample_data = np.array([[
            17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871,
            1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193,
            25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189
        ]])
        
        # Scale and predict
        sample_scaled = scaler.transform(sample_data)
        prediction = model.predict(sample_scaled)
        probabilities = model.predict_proba(sample_scaled)
        
        print(f"Test prediction: {prediction[0]} ({'Benign' if prediction[0] == 1 else 'Malignant'})")
        print(f"Probabilities: Malignant={probabilities[0][0]:.4f}, Benign={probabilities[0][1]:.4f}")
        print("Model verification successful!")
        
        return True
        
    except Exception as e:
        print(f"Model verification failed: {str(e)}")
        return False

def main():
    """
    Main function to create and verify the model.
    """
    try:
        # Create the model
        model_path = create_model()
        
        # Verify the model
        if verify_model(model_path):
            print("\n✅ Model creation completed successfully!")
            print(f"You can now use the model at: {model_path}")
        else:
            print("\n❌ Model verification failed!")
            
    except Exception as e:
        print(f"Error creating model: {str(e)}")

if __name__ == "__main__":
    main()