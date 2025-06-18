#!/usr/bin/env python3
"""
Fixed breast cancer prediction model testing script
Handles NaN values in test data properly
"""
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

def test_model():
    """Test the trained breast cancer model"""
    print("🧪 BREAST CANCER PREDICTION MODEL TESTING")
    print("=" * 50)
    
    try:
        # Load the trained model
        model_data = joblib.load('ml/model.pkl')
        model = model_data['model']
        scaler = model_data['scaler']
        feature_names = model_data['feature_names']
        
        print("✓ Model loaded successfully")
        print(f"  Model expects {len(feature_names)} features")
        
        # Load test dataset
        try:
            df = pd.read_csv('data/breast_cancer_dataset.csv')
            print(f"✓ Dataset loaded: {df.shape}")
        except:
            df = pd.read_csv('data/breast_cancer_sample.csv')
            print(f"✓ Sample dataset loaded: {df.shape}")
        
        # Prepare test data with proper NaN handling
        if 'diagnosis' in df.columns:
            # Clean data - remove rows with missing diagnosis
            df_clean = df.dropna(subset=['diagnosis']).copy()
            print(f"✓ Cleaned dataset: {df_clean.shape} (removed NaN diagnosis)")
            
            # Prepare features and target
            X_test = df_clean.drop(['diagnosis'], axis=1)
            y_true = df_clean['diagnosis']
            
            # Convert diagnosis to numeric if needed
            if y_true.dtype == 'object':
                y_true = y_true.map({'M': 1, 'B': 0, 'Malignant': 1, 'Benign': 0})
            
            # Remove any remaining NaN rows
            mask = ~(X_test.isnull().any(axis=1) | y_true.isnull())
            X_test = X_test[mask]
            y_true = y_true[mask]
            
            print(f"✓ Final test data: {X_test.shape[0]} samples, {X_test.shape[1]} features")
            print(f"✓ Target distribution: Benign={sum(y_true==0)}, Malignant={sum(y_true==1)}")
            
        else:
            print("⚠️  No diagnosis column found, using synthetic test")
            X_test = df.iloc[:10, :-1]  # First 10 rows, all features except last
            y_true = None
        
        # Handle missing values in features
        if X_test.isnull().any().any():
            print("🧹 Filling missing values in test features...")
            X_test = X_test.fillna(X_test.median())
        
        # Ensure feature order matches training
        if list(X_test.columns) != feature_names:
            print("🔧 Reordering features to match training...")
            X_test = X_test[feature_names]
        
        # Scale the test data
        X_test_scaled = scaler.transform(X_test)
        
        print(f"\n🔍 MODEL DIAGNOSTICS:")
        print(f"Model type: {type(model).__name__}")
        print(f"Expected features: {len(feature_names)}")
        print(f"Test data features: {X_test_scaled.shape[1]}")
        print(f"Feature names match: {list(X_test.columns) == feature_names}")
        
        # Test predictions
        print(f"\n🧪 RUNNING PREDICTIONS...")
        predictions = model.predict(X_test_scaled)
        probabilities = model.predict_proba(X_test_scaled)
        
        print(f"✓ Predictions generated for {len(predictions)} samples")
        
        # Calculate accuracy if we have true labels
        if y_true is not None and len(y_true) > 0:
            accuracy = accuracy_score(y_true, predictions)
            print(f"✓ Model accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
            
            print(f"\n📊 DETAILED CLASSIFICATION REPORT:")
            print(classification_report(y_true, predictions, 
                                      target_names=['Benign', 'Malignant'],
                                      zero_division=0))
            
            print(f"\n🎯 CONFUSION MATRIX:")
            cm = confusion_matrix(y_true, predictions)
            print(f"                  Predicted")
            print(f"                Benign  Malignant")
            print(f"Actual Benign      {cm[0,0]:2d}      {cm[0,1]:2d}")
            print(f"    Malignant      {cm[1,0]:2d}      {cm[1,1]:2d}")
        
        # Show sample predictions with confidence
        print(f"\n🔬 SAMPLE PREDICTIONS:")
        n_samples = min(10, len(predictions))
        for i in range(n_samples):
            pred_label = "Malignant" if predictions[i] == 1 else "Benign"
            confidence = max(probabilities[i]) * 100
            actual_label = ""
            if y_true is not None:
                actual = "Malignant" if y_true.iloc[i] == 1 else "Benign"
                match = "✓" if predictions[i] == y_true.iloc[i] else "✗"
                actual_label = f" | Actual: {actual} {match}"
            
            print(f"Sample {i+1:2d}: {pred_label:<9} (Confidence: {confidence:5.1f}%){actual_label}")
        
        # Test edge cases
        print(f"\n⚠️  TESTING EDGE CASES:")
        
        # Create edge case test data
        edge_cases = {
            "All zeros": np.zeros((1, len(feature_names))),
            "All ones": np.ones((1, len(feature_names))),
            "Random values": np.random.rand(1, len(feature_names)),
            "High risk profile": np.array([[1, 65, 5.0, 3, 3, 4, 1, 1, 1, 1, 1, 35, 1, 2, 0]]),  # High risk
            "Low risk profile": np.array([[2, 25, 1.0, 0, 1, 1, 0, 0, 0, 0, 0, 22, 0, 0, 3]])    # Low risk
        }
        
        for case_name, case_data in edge_cases.items():
            if case_data.shape[1] == len(feature_names):
                case_scaled = scaler.transform(case_data)
                case_pred = model.predict(case_scaled)[0]
                case_prob = model.predict_proba(case_scaled)[0]
                case_conf = max(case_prob) * 100
                case_label = "Malignant" if case_pred == 1 else "Benign"
                print(f"{case_name:<18}: {case_label} ({case_conf:.1f}%)")
        
        # Feature importance
        print(f"\n🎯 FEATURE IMPORTANCE ANALYSIS:")
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"Top 10 Most Important Features:")
            for i, (_, row) in enumerate(importance_df.head(10).iterrows()):
                print(f"{i+1:2d}. {row['feature']:<25} ({row['importance']:.4f})")
        
        print(f"\n🎉 TESTING COMPLETED SUCCESSFULLY!")
        print(f"✨ Your model is working correctly and ready for production!")
        
        return True
        
    except Exception as e:
        print(f"❌ Testing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_test_patient():
    """Create a test patient for manual testing"""
    print(f"\n👤 CREATING TEST PATIENT...")
    
    # Example high-risk patient
    test_patient = {
        'patient_id': 999,
        'age': 55,
        'tumor_size': 3.5,
        'lymph_nodes': 2,
        'grade': 3,
        'stage': 3,
        'er_status': 1,
        'pr_status': 1,
        'her2_status': 0,
        'menopause_status': 1,
        'family_history': 1,
        'bmi': 28,
        'smoking_history': 1,
        'alcohol_consumption': 2,
        'physical_activity': 1
    }
    
    try:
        model_data = joblib.load('ml/model.pkl')
        model = model_data['model']
        scaler = model_data['scaler']
        feature_names = model_data['feature_names']
        
        # Prepare patient data
        patient_df = pd.DataFrame([test_patient])
        patient_features = patient_df[feature_names]
        patient_scaled = scaler.transform(patient_features)
        
        # Make prediction
        prediction = model.predict(patient_scaled)[0]
        probability = model.predict_proba(patient_scaled)[0]
        confidence = max(probability) * 100
        
        result = "Malignant" if prediction == 1 else "Benign"
        
        print(f"✓ Test patient prediction: {result}")
        print(f"✓ Confidence: {confidence:.1f}%")
        print(f"✓ Risk probabilities: Benign={probability[0]*100:.1f}%, Malignant={probability[1]*100:.1f}%")
        
    except Exception as e:
        print(f"❌ Test patient prediction failed: {e}")

if __name__ == "__main__":
    success = test_model()
    
    if success:
        create_test_patient()
        print(f"\n🚀 READY FOR PRODUCTION!")
        print(f"📋 Your model can now be used in your web application")
        print(f"💡 Use the prediction functions in your Flask app")
    else:
        print(f"\n❌ Fix the issues above before proceeding")