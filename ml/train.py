import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

def load_data():
    """
    Load breast cancer dataset.
    In a real scenario, you would load your actual dataset here.
    For demonstration, we'll use sklearn's built-in breast cancer dataset.
    """
    from sklearn.datasets import load_breast_cancer
    
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name='target')
    
    return X, y

def preprocess_data(X, y):
    """
    Preprocess the data for training.
    """
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def train_model(X_train, y_train):
    """
    Train the Random Forest model.
    """
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2
    )
    
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model.
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Model Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    return accuracy

def save_model(model, scaler, filename='model.pkl'):
    """
    Save the trained model and scaler to a pickle file.
    """
    model_data = {
        'model': model,
        'scaler': scaler,
        'feature_names': None  # Will be updated if needed
    }
    
    # Create ml directory if it doesn't exist
    os.makedirs('ml', exist_ok=True)
    
    filepath = os.path.join('ml', filename)
    joblib.dump(model_data, filepath)
    print(f"Model saved to {filepath}")

def main():
    """
    Main training pipeline.
    """
    print("Starting breast cancer prediction model training...")
    
    # Load data
    print("Loading data...")
    X, y = load_data()
    print(f"Dataset shape: {X.shape}")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    
    # Preprocess data
    print("Preprocessing data...")
    X_train, X_test, y_train, y_test, scaler = preprocess_data(X, y)
    
    # Train model
    print("Training model...")
    model = train_model(X_train, y_train)
    
    # Evaluate model
    print("Evaluating model...")
    accuracy = evaluate_model(model, X_test, y_test)
    
    # Save model
    print("Saving model...")
    save_model(model, scaler)
    
    print(f"\nTraining completed! Final accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()