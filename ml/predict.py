import numpy as np
import pandas as pd
import joblib
import os
from typing import Dict, List, Union

class BreastCancerPredictor:
    """
    Breast Cancer Prediction class for making predictions using the trained model.
    """
    
    def __init__(self, model_path: str = 'ml/model.pkl'):
        self.model_path = model_path
        self.model_data = None
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.load_model()
    
    def load_model(self):
        """
        Load the trained model and scaler from pickle file.
        """
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at {self.model_path}")
        
        try:
            self.model_data = joblib.load(self.model_path)
            self.model = self.model_data['model']
            self.scaler = self.model_data['scaler']
            self.feature_names = self.model_data.get('feature_names', None)
            print("Model loaded successfully!")
        except Exception as e:
            raise Exception(f"Error loading model: {str(e)}")
    
    def validate_input(self, features: Union[Dict, List, np.ndarray]) -> np.ndarray:
        """
        Validate and preprocess input features.
        
        Args:
            features: Input features as dict, list, or numpy array
            
        Returns:
            Processed numpy array ready for prediction
        """
        if isinstance(features, dict):
            # Convert dictionary to array (assuming ordered features)
            features = list(features.values())
        
        if isinstance(features, list):
            features = np.array(features)
        
        # Ensure it's a 2D array
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # Validate feature count (breast cancer dataset has 30 features)
        expected_features = 30
        if features.shape[1] != expected_features:
            raise ValueError(f"Expected {expected_features} features, got {features.shape[1]}")
        
        return features
    
    def predict(self, features: Union[Dict, List, np.ndarray]) -> Dict:
        """
        Make prediction on input features.
        
        Args:
            features: Input features
            
        Returns:
            Dictionary containing prediction results
        """
        try:
            # Validate and preprocess input
            processed_features = self.validate_input(features)
            
            # Scale features
            scaled_features = self.scaler.transform(processed_features)
            
            # Make prediction
            prediction = self.model.predict(scaled_features)[0]
            prediction_proba = self.model.predict_proba(scaled_features)[0]
            
            # Prepare result
            result = {
                'prediction': int(prediction),
                'prediction_label': 'Malignant' if prediction == 0 else 'Benign',
                'confidence': float(max(prediction_proba)),
                'probabilities': {
                    'malignant': float(prediction_proba[0]),
                    'benign': float(prediction_proba[1])
                },
                'risk_level': self._get_risk_level(prediction_proba)
            }
            
            return result
            
        except Exception as e:
            return {'error': str(e)}
    
    def predict_batch(self, features_list: List[Union[Dict, List, np.ndarray]]) -> List[Dict]:
        """
        Make predictions on multiple samples.
        
        Args:
            features_list: List of feature arrays/dicts
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        for features in features_list:
            result = self.predict(features)
            results.append(result)
        return results
    
    def _get_risk_level(self, probabilities: np.ndarray) -> str:
        """
        Determine risk level based on prediction probabilities.
        
        Args:
            probabilities: Array of class probabilities
            
        Returns:
            Risk level string
        """
        malignant_prob = probabilities[0]
        
        if malignant_prob >= 0.8:
            return 'High Risk'
        elif malignant_prob >= 0.5:
            return 'Moderate Risk'
        elif malignant_prob >= 0.3:
            return 'Low Risk'
        else:
            return 'Very Low Risk'
    
    def get_feature_importance(self, top_n: int = 10) -> Dict:
        """
        Get feature importance from the trained model.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            Dictionary with feature importance information
        """
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
            
            # Create feature names if not available
            if self.feature_names is None:
                feature_names = [f'feature_{i}' for i in range(len(importance))]
            else:
                feature_names = self.feature_names
            
            # Sort by importance
            importance_pairs = list(zip(feature_names, importance))
            importance_pairs.sort(key=lambda x: x[1], reverse=True)
            
            return {
                'top_features': importance_pairs[:top_n],
                'all_features': importance_pairs
            }
        else:
            return {'error': 'Model does not support feature importance'}

def create_sample_features():
    """
    Create sample features for testing.
    This represents the 30 features of the breast cancer dataset.
    """
    # Sample mean values from the breast cancer dataset
    sample_features = [
        14.127292, 19.289649, 91.969033, 654.889104, 0.096360,
        0.104341, 0.088799, 0.048919, 0.181162, 0.062798,
        0.405172, 1.216853, 2.866059, 40.337079, 0.007041,
        0.025478, 0.031894, 0.011796, 0.020542, 0.003795,
        16.269190, 25.677223, 107.261213, 880.583128, 0.132369,
        0.254265, 0.272188, 0.114606, 0.290076, 0.083946
    ]
    
    return sample_features

# Example usage and testing functions
def main():
    """
    Main function for testing the predictor.
    """
    try:
        # Initialize predictor
        predictor = BreastCancerPredictor()
        
        # Create sample features
        sample_features = create_sample_features()
        
        # Make prediction
        result = predictor.predict(sample_features)
        
        print("Prediction Result:")
        print(f"Prediction: {result.get('prediction_label', 'Unknown')}")
        print(f"Confidence: {result.get('confidence', 0):.4f}")
        print(f"Risk Level: {result.get('risk_level', 'Unknown')}")
        print(f"Probabilities: Malignant={result['probabilities']['malignant']:.4f}, "
              f"Benign={result['probabilities']['benign']:.4f}")
        
        # Get feature importance
        importance = predictor.get_feature_importance(top_n=5)
        if 'top_features' in importance:
            print("\nTop 5 Important Features:")
            for feature, imp in importance['top_features']:
                print(f"  {feature}: {imp:.4f}")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()