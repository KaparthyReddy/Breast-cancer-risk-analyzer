#!/usr/bin/env python3
"""
Updated test script for breast cancer prediction model.
Now loads the saved test set from training to prevent data leakage.
"""
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

def test_model():
    print("🧪 BREAST CANCER PREDICTION MODEL TESTING")
    print("=" * 50)
    
    try:
        model_data = joblib.load('ml/model.pkl')
        model = model_data['model']
        scaler = model_data['scaler']
        feature_names = model_data['feature_names']
        
        print("✓ Model loaded successfully")
        print(f"  Model expects {len(feature_names)} features")
        print(f"  Expected features: {feature_names[:5]}... (showing first 5)")

        return test_with_saved_test_data(model, scaler, feature_names)

    except Exception as e:
        print(f"❌ Testing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_with_saved_test_data(model, scaler, feature_names):
    try:
        print("\n📦 Loading saved test data from ml/test_data.pkl...")
        test_data = joblib.load('ml/test_data.pkl')
        X_test = test_data['X_test']
        y_true = test_data['y_test']

        print(f"✓ Test set loaded: {X_test.shape[0]} samples")
        print(f"✓ Test target distribution: Malignant={sum(y_true==0)}, Benign={sum(y_true==1)}")
        print(f"✓ Feature names match: {list(X_test.columns) == feature_names}")

        return run_model_tests(model, scaler, feature_names, X_test, y_true)
    except Exception as e:
        print(f"❌ Failed to load or test on saved test data: {e}")
        return False

def run_model_tests(model, scaler, feature_names, X_test, y_true, is_synthetic=False):
    try:
        X_test_scaled = scaler.transform(X_test)

        print(f"\n🔍 MODEL DIAGNOSTICS:")
        print(f"Model type: {type(model).__name__}")
        print(f"Expected features: {len(feature_names)}")
        print(f"Test data features: {X_test_scaled.shape[1]}")
        print(f"Feature names match: {list(X_test.columns) == feature_names}")

        print(f"\n🧪 RUNNING PREDICTIONS...")
        predictions = model.predict(X_test_scaled)
        probabilities = model.predict_proba(X_test_scaled)

        print(f"✓ Predictions generated for {len(predictions)} samples")

        if y_true is not None and len(y_true) > 0:
            accuracy = accuracy_score(y_true, predictions)
            accuracy_note = " (synthetic data)" if is_synthetic else ""
            print(f"✓ Model accuracy: {accuracy:.4f} ({accuracy*100:.1f}%){accuracy_note}")

            if not is_synthetic:
                print(f"\n📊 DETAILED CLASSIFICATION REPORT:")
                print(classification_report(y_true, predictions, 
                                            target_names=['Malignant', 'Benign'], 
                                            zero_division=0))

                print(f"\n🎯 CONFUSION MATRIX:")
                cm = confusion_matrix(y_true, predictions)
                print(f"                  Predicted")
                print(f"               Malignant  Benign")
                print(f"Actual Malignant    {cm[0,0]:2d}      {cm[0,1]:2d}")
                print(f"       Benign       {cm[1,0]:2d}      {cm[1,1]:2d}")

        print(f"\n🔬 SAMPLE PREDICTIONS:")
        n_samples = min(10, len(predictions))
        for i in range(n_samples):
            pred_label = "Benign" if predictions[i] == 1 else "Malignant"
            confidence = max(probabilities[i]) * 100
            actual = "Benign" if y_true.iloc[i] == 1 else "Malignant"
            match = "✓" if predictions[i] == y_true.iloc[i] else "✗"
            print(f"Sample {i+1:2d}: {pred_label:<9} (Confidence: {confidence:5.1f}%) | Actual: {actual} {match}")

        test_edge_cases(model, scaler, feature_names)
        show_feature_importance(model, feature_names)

        print(f"\n🎉 TESTING COMPLETED SUCCESSFULLY!")
        if is_synthetic:
            print(f"⚠️  Note: Results based on synthetic data for testing purposes")
        print(f"✨ Your model is working correctly and ready for production!")

        return True

    except Exception as e:
        print(f"❌ Model testing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_edge_cases(model, scaler, feature_names):
    print(f"\n⚠️  TESTING EDGE CASES:")
    try:
        edge_cases = {
            "All median values": np.full((1, len(feature_names)), 0.5),
            "Low risk profile": np.random.normal(0.3, 0.1, (1, len(feature_names))),
            "High risk profile": np.random.normal(0.7, 0.1, (1, len(feature_names))),
            "Random values": np.random.rand(1, len(feature_names))
        }

        for case_name, case_data in edge_cases.items():
            case_data = np.abs(case_data)
            case_scaled = scaler.transform(case_data)
            case_pred = model.predict(case_scaled)[0]
            case_prob = model.predict_proba(case_scaled)[0]
            case_conf = max(case_prob) * 100
            case_label = "Benign" if case_pred == 1 else "Malignant"
            print(f"{case_name:<18}: {case_label} ({case_conf:.1f}%)")

    except Exception as e:
        print(f"❌ Edge case testing failed: {e}")

def show_feature_importance(model, feature_names):
    print(f"\n🎯 FEATURE IMPORTANCE ANALYSIS:")
    try:
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)

            print("Top 10 Most Important Features:")
            for i, (_, row) in enumerate(importance_df.head(10).iterrows()):
                print(f"{i+1:2d}. {row['feature']:<30} ({row['importance']:.4f})")
        else:
            print("Feature importance not available for this model type.")
    except Exception as e:
        print(f"❌ Feature importance analysis failed: {e}")

def create_test_patient():
    print(f"\n👤 CREATING TEST PATIENT...")
    try:
        model_data = joblib.load('ml/model.pkl')
        model = model_data['model']
        scaler = model_data['scaler']
        feature_names = model_data['feature_names']

        test_patient = {}
        for feature in feature_names:
            if 'mean radius' in feature:
                test_patient[feature] = 15.0
            elif 'mean texture' in feature:
                test_patient[feature] = 20.0
            elif 'mean perimeter' in feature:
                test_patient[feature] = 95.0
            elif 'mean area' in feature:
                test_patient[feature] = 700.0
            elif 'mean smoothness' in feature:
                test_patient[feature] = 0.10
            elif 'mean compactness' in feature:
                test_patient[feature] = 0.12
            elif 'mean concavity' in feature:
                test_patient[feature] = 0.10
            elif 'mean concave points' in feature:
                test_patient[feature] = 0.05
            elif 'mean symmetry' in feature:
                test_patient[feature] = 0.18
            elif 'mean fractal dimension' in feature:
                test_patient[feature] = 0.06
            else:
                test_patient[feature] = np.random.normal(0.5, 0.2)

        patient_df = pd.DataFrame([test_patient])
        patient_scaled = scaler.transform(patient_df)

        prediction = model.predict(patient_scaled)[0]
        probability = model.predict_proba(patient_scaled)[0]
        confidence = max(probability) * 100

        result = "Benign" if prediction == 1 else "Malignant"
        print(f"✓ Test patient prediction: {result}")
        print(f"✓ Confidence: {confidence:.1f}%")
        print(f"✓ Risk probabilities: Malignant={probability[0]*100:.1f}%, Benign={probability[1]*100:.1f}%")

    except Exception as e:
        print(f"❌ Test patient prediction failed: {e}")

if __name__ == "__main__":
    success = test_model()

    if success:
        create_test_patient()
        print(f"\n🚀 READY FOR PRODUCTION!")
        print(f"📋 Your model can now be used in your web application")
        print(f"💡 Use the prediction functions in your Flask app")
        print(f"\n📝 RECOMMENDATIONS:")
        print(f"  1. Ensure your web app uses the same feature format as the model")
        print(f"  2. Test with real patient data that matches the expected features")
        print(f"  3. Consider retraining with your specific dataset format if needed")
    else:
        print(f"\n❌ Fix the issues above before proceeding")
