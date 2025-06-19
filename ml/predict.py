#!/usr/bin/env python3
"""
Enhanced Breast Cancer Predictor with Advanced Feature Engineering
Supports the new enhanced model with feature engineering and ensemble methods
"""

import numpy as np
import pandas as pd
import joblib
import os
from typing import Dict, List, Union, Tuple
import warnings
warnings.filterwarnings('ignore')

class EnhancedBreastCancerPredictor:
    """
    Enhanced Breast Cancer Prediction class with advanced feature engineering
    and ensemble model support.
    """
    
    def __init__(self, model_path: str = 'ml/enhanced_model.pkl'):
        self.model_path = model_path
        self.model_data = None
        self.model = None
        self.scaler = None
        self.selector = None
        self.feature_names = None
        self.original_feature_names = None
        self.load_model()
    
    def load_model(self):
        """Load the enhanced trained model and preprocessing components."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at {self.model_path}")
        
        try:
            self.model_data = joblib.load(self.model_path)
            self.model = self.model_data['model']
            self.scaler = self.model_data['scaler']
            self.selector = self.model_data.get('selector', None)
            self.feature_names = self.model_data.get('feature_names', None)
            
            # Original 30 breast cancer features
            self.original_feature_names = [
                'mean radius', 'mean texture', 'mean perimeter', 'mean area',
                'mean smoothness', 'mean compactness', 'mean concavity',
                'mean concave points', 'mean symmetry', 'mean fractal dimension',
                'radius error', 'texture error', 'perimeter error', 'area error',
                'smoothness error', 'compactness error', 'concavity error',
                'concave points error', 'symmetry error', 'fractal dimension error',
                'worst radius', 'worst texture', 'worst perimeter', 'worst area',
                'worst smoothness', 'worst compactness', 'worst concavity',
                'worst concave points', 'worst symmetry', 'worst fractal dimension'
            ]
            
            print("Enhanced model loaded successfully!")
            print(f"Model type: {self.model_data.get('model_type', 'Unknown')}")
            print(f"Feature count: {self.model_data.get('feature_count', 'Unknown')}")
            
        except Exception as e:
            raise Exception(f"Error loading model: {str(e)}")
    
    def advanced_feature_engineering(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the same advanced feature engineering used in training."""
        X_enhanced = X.copy()
        
        # Original ratio features
        X_enhanced['mean_area_to_perimeter_ratio'] = X['mean area'] / (X['mean perimeter'] + 1e-8)
        X_enhanced['mean_compactness_to_concavity_ratio'] = X['mean compactness'] / (X['mean concavity'] + 1e-8)
        X_enhanced['texture_smoothness_interaction'] = X['mean texture'] * X['mean smoothness']
        
        # Advanced mathematical features
        X_enhanced['perimeter_area_ratio'] = X['mean perimeter'] / np.sqrt(X['mean area'] + 1e-8)
        X_enhanced['concavity_compactness_product'] = X['mean concavity'] * X['mean compactness']
        X_enhanced['symmetry_fractal_interaction'] = X['mean symmetry'] * X['mean fractal dimension']
        
        # Statistical features across mean, SE, worst
        feature_groups = [
            ['mean radius', 'radius error', 'worst radius'],
            ['mean texture', 'texture error', 'worst texture'],
            ['mean perimeter', 'perimeter error', 'worst perimeter'],
            ['mean area', 'area error', 'worst area'],
            ['mean smoothness', 'smoothness error', 'worst smoothness'],
            ['mean compactness', 'compactness error', 'worst compactness'],
            ['mean concavity', 'concavity error', 'worst concavity'],
            ['mean concave points', 'concave points error', 'worst concave points'],
            ['mean symmetry', 'symmetry error', 'worst symmetry'],
            ['mean fractal dimension', 'fractal dimension error', 'worst fractal dimension']
        ]
        
        for group in feature_groups:
            if all(col in X.columns for col in group):
                base_name = group[0].replace('mean ', '').replace(' ', '_')
                X_enhanced[f'{base_name}_range'] = X[group[2]] - X[group[0]]  # worst - mean
                X_enhanced[f'{base_name}_cv'] = X[group[1]] / (X[group[0]] + 1e-8)  # CV
                X_enhanced[f'{base_name}_severity'] = X[group[2]] / (X[group[0]] + 1e-8)  # severity ratio
        
        # Polynomial features for most important features
        top_features = ['worst concave points', 'worst perimeter', 'worst radius', 
                       'mean concave points', 'worst area', 'mean concavity']
        
        for feature in top_features:
            if feature in X.columns:
                X_enhanced[f'{feature}_squared'] = X[feature] ** 2
                X_enhanced[f'{feature}_cubed'] = X[feature] ** 3
                X_enhanced[f'{feature}_log'] = np.log1p(X[feature])
                X_enhanced[f'{feature}_sqrt'] = np.sqrt(X[feature])
        
        # Cross-feature interactions
        X_enhanced['worst_concave_points_x_perimeter'] = X['worst concave points'] * X['worst perimeter']
        X_enhanced['mean_concave_points_x_area'] = X['mean concave points'] * X['mean area']
        X_enhanced['compactness_concavity_worst'] = X['worst compactness'] * X['worst concavity']
        
        # Remove any infinite or NaN values
        X_enhanced = X_enhanced.replace([np.inf, -np.inf], np.nan)
        X_enhanced = X_enhanced.fillna(X_enhanced.median())
        
        return X_enhanced
    
    def validate_input(self, features: Union[Dict, List, np.ndarray]) -> pd.DataFrame:
        """Validate and preprocess input features."""
        if isinstance(features, dict):
            # Ensure all required features are present
            if len(features) != 30:
                raise ValueError(f"Expected 30 features, got {len(features)}")
            # Convert to DataFrame
            df = pd.DataFrame([features])
        elif isinstance(features, list):
            if len(features) != 30:
                raise ValueError(f"Expected 30 features, got {len(features)}")
            # Convert to DataFrame with proper column names
            df = pd.DataFrame([features], columns=self.original_feature_names)
        elif isinstance(features, np.ndarray):
            if features.ndim == 1:
                features = features.reshape(1, -1)
            if features.shape[1] != 30:
                raise ValueError(f"Expected 30 features, got {features.shape[1]}")
            # Convert to DataFrame
            df = pd.DataFrame(features, columns=self.original_feature_names)
        else:
            raise ValueError("Features must be dict, list, or numpy array")
        
        return df
    
    def preprocess_features(self, df: pd.DataFrame) -> np.ndarray:
        """Apply feature engineering and preprocessing pipeline."""
        # Apply advanced feature engineering
        df_enhanced = self.advanced_feature_engineering(df)
        
        # Ensure all expected features are present (fill missing with median)
        if self.feature_names:
            for feature in self.feature_names:
                if feature not in df_enhanced.columns:
                    df_enhanced[feature] = 0.0  # Default value
            
            # Reorder columns to match training order
            df_enhanced = df_enhanced[self.feature_names]
        
        # Convert to numpy array
        features_array = df_enhanced.values
        
        # Apply feature selection if used during training
        if self.selector is not None:
            features_array = self.selector.transform(features_array)
        
        # Apply scaling
        if self.scaler is not None:
            features_array = self.scaler.transform(features_array)
        
        return features_array
    
    def predict(self, features: Union[Dict, List, np.ndarray]) -> Dict:
        """Make prediction on input features."""
        try:
            # Validate and preprocess input
            df = self.validate_input(features)
            processed_features = self.preprocess_features(df)
            
            # Make prediction
            prediction = self.model.predict(processed_features)[0]
            prediction_proba = self.model.predict_proba(processed_features)[0]
            
            # Calculate confidence and risk metrics
            confidence = float(max(prediction_proba))
            malignant_prob = float(prediction_proba[0])
            benign_prob = float(prediction_proba[1])
            
            # Enhanced risk assessment
            risk_level = self._get_enhanced_risk_level(malignant_prob)
            risk_score = self._calculate_risk_score(malignant_prob)
            
            # Prepare comprehensive result
            result = {
                'prediction': int(prediction),
                'prediction_label': 'Malignant' if prediction == 0 else 'Benign',
                'confidence': confidence,
                'probability': malignant_prob,  # For backward compatibility with Flask app
                'probabilities': {
                    'malignant': malignant_prob,
                    'benign': benign_prob
                },
                'risk_level': risk_level,
                'risk_score': risk_score,
                'interpretation': self._get_interpretation(prediction, malignant_prob),
                'recommendations': self._get_recommendations(risk_level, malignant_prob)
            }
            
            return result
            
        except Exception as e:
            return {'error': str(e)}
    
    def predict_batch(self, features_list: List[Union[Dict, List, np.ndarray]]) -> List[Dict]:
        """Make predictions on multiple samples."""
        results = []
        for i, features in enumerate(features_list):
            try:
                result = self.predict(features)
                result['sample_id'] = i
                results.append(result)
            except Exception as e:
                results.append({'sample_id': i, 'error': str(e)})
        return results
    
    def _get_enhanced_risk_level(self, malignant_prob: float) -> str:
        """Enhanced risk level assessment with more granular categories."""
        if malignant_prob >= 0.9:
            return 'Very High Risk'
        elif malignant_prob >= 0.7:
            return 'High Risk'
        elif malignant_prob >= 0.5:
            return 'Moderate Risk'
        elif malignant_prob >= 0.3:
            return 'Low Risk'
        elif malignant_prob >= 0.1:
            return 'Very Low Risk'
        else:
            return 'Minimal Risk'
    
    def _calculate_risk_score(self, malignant_prob: float) -> int:
        """Calculate a risk score from 0-100."""
        return int(malignant_prob * 100)
    
    def _get_interpretation(self, prediction: int, malignant_prob: float) -> str:
        """Provide interpretation of the prediction."""
        if prediction == 0:  # Malignant
            if malignant_prob >= 0.9:
                return "High confidence malignant prediction - immediate medical attention recommended"
            elif malignant_prob >= 0.7:
                return "Likely malignant - urgent medical consultation advised"
            else:
                return "Possible malignant - medical evaluation recommended"
        else:  # Benign
            if malignant_prob <= 0.1:
                return "High confidence benign prediction - routine monitoring sufficient"
            elif malignant_prob <= 0.3:
                return "Likely benign - standard follow-up recommended"
            else:
                return "Probably benign - closer monitoring may be advisable"
    
    def _get_recommendations(self, risk_level: str, malignant_prob: float) -> List[str]:
        """Provide recommendations based on risk level."""
        recommendations = []
        
        if risk_level in ['Very High Risk', 'High Risk']:
            recommendations.extend([
                "Seek immediate medical attention",
                "Request urgent oncology consultation",
                "Consider additional imaging studies",
                "Discuss biopsy options with physician"
            ])
        elif risk_level == 'Moderate Risk':
            recommendations.extend([
                "Schedule medical consultation soon",
                "Consider follow-up imaging in 3-6 months",
                "Discuss family history with doctor",
                "Maintain regular screening schedule"
            ])
        else:
            recommendations.extend([
                "Continue routine screening",
                "Maintain healthy lifestyle",
                "Be aware of any changes",
                "Follow standard screening guidelines"
            ])
        
        return recommendations
    
    def get_feature_importance(self, top_n: int = 15) -> Dict:
        """Get feature importance from the trained model."""
        try:
            # Handle different model types
            if hasattr(self.model, 'feature_importances_'):
                importance = self.model.feature_importances_
            elif hasattr(self.model, 'estimators_'):
                # For ensemble models, try to get feature importance
                if hasattr(self.model.estimators_[0], 'feature_importances_'):
                    # Average importance across estimators
                    importances = [est.feature_importances_ for est in self.model.estimators_]
                    importance = np.mean(importances, axis=0)
                else:
                    return {'error': 'Model does not support feature importance'}
            else:
                return {'error': 'Model does not support feature importance'}
            
            # Get feature names after selection
            if self.selector is not None and self.feature_names:
                selected_features = np.array(self.feature_names)[self.selector.get_support()]
            elif self.feature_names:
                selected_features = self.feature_names
            else:
                selected_features = [f'feature_{i}' for i in range(len(importance))]
            
            # Sort by importance
            importance_pairs = list(zip(selected_features, importance))
            importance_pairs.sort(key=lambda x: x[1], reverse=True)
            
            return {
                'top_features': importance_pairs[:top_n],
                'all_features': importance_pairs,
                'total_features': len(importance_pairs)
            }
            
        except Exception as e:
            return {'error': f'Error getting feature importance: {str(e)}'}
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model."""
        return {
            'model_type': self.model_data.get('model_type', 'Unknown'),
            'feature_count': self.model_data.get('feature_count', 'Unknown'),
            'has_feature_selector': self.selector is not None,
            'scaler_type': type(self.scaler).__name__ if self.scaler else 'None',
            'original_features': len(self.original_feature_names),
            'engineered_features': len(self.feature_names) if self.feature_names else 'Unknown'
        }

def create_sample_features() -> Dict:
    """Create sample features for testing (malignant case)."""
    feature_names = [
        'mean radius', 'mean texture', 'mean perimeter', 'mean area',
        'mean smoothness', 'mean compactness', 'mean concavity',
        'mean concave points', 'mean symmetry', 'mean fractal dimension',
        'radius error', 'texture error', 'perimeter error', 'area error',
        'smoothness error', 'compactness error', 'concavity error',
        'concave points error', 'symmetry error', 'fractal dimension error',
        'worst radius', 'worst texture', 'worst perimeter', 'worst area',
        'worst smoothness', 'worst compactness', 'worst concavity',
        'worst concave points', 'worst symmetry', 'worst fractal dimension'
    ]
    
    # Sample malignant case values
    malignant_values = [
        17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871,
        1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193,
        25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189
    ]
    
    return dict(zip(feature_names, malignant_values))

def create_benign_sample_features() -> Dict:
    """Create sample features for testing (benign case)."""
    feature_names = [
        'mean radius', 'mean texture', 'mean perimeter', 'mean area',
        'mean smoothness', 'mean compactness', 'mean concavity',
        'mean concave points', 'mean symmetry', 'mean fractal dimension',
        'radius error', 'texture error', 'perimeter error', 'area error',
        'smoothness error', 'compactness error', 'concavity error',
        'concave points error', 'symmetry error', 'fractal dimension error',
        'worst radius', 'worst texture', 'worst perimeter', 'worst area',
        'worst smoothness', 'worst compactness', 'worst concavity',
        'worst concave points', 'worst symmetry', 'worst fractal dimension'
    ]
    
    # Sample benign case values
    benign_values = [
        13.54, 14.36, 87.46, 566.3, 0.09779, 0.08129, 0.06664, 0.04781, 0.1885, 0.05766,
        0.2699, 0.7886, 2.058, 23.56, 0.008462, 0.0146, 0.02387, 0.01315, 0.0198, 0.0023,
        15.11, 19.26, 99.7, 711.2, 0.144, 0.1773, 0.239, 0.1288, 0.2977, 0.07259
    ]
    
    return dict(zip(feature_names, benign_values))

def main():
    """Main function for testing the enhanced predictor."""
    try:
        print("Enhanced Breast Cancer Predictor - Testing")
        print("=" * 50)
        
        # Initialize predictor
        predictor = EnhancedBreastCancerPredictor()
        
        # Display model information
        info = predictor.get_model_info()
        print(f"Model Information:")
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        print("\n" + "=" * 50)
        
        # Test with malignant sample
        print("Testing with Malignant Sample:")
        print("-" * 30)
        malignant_features = create_sample_features()
        result = predictor.predict(malignant_features)
        
        if 'error' not in result:
            print(f"Prediction: {result['prediction_label']}")
            print(f"Confidence: {result['confidence']:.4f}")
            print(f"Risk Level: {result['risk_level']}")
            print(f"Risk Score: {result['risk_score']}/100")
            print(f"Probabilities: Malignant={result['probabilities']['malignant']:.4f}, "
                  f"Benign={result['probabilities']['benign']:.4f}")
            print(f"Interpretation: {result['interpretation']}")
            print("Recommendations:")
            for rec in result['recommendations']:
                print(f"  - {rec}")
        else:
            print(f"Error: {result['error']}")
        
        print("\n" + "=" * 50)
        
        # Test with benign sample
        print("Testing with Benign Sample:")
        print("-" * 30)
        benign_features = create_benign_sample_features()
        result = predictor.predict(benign_features)
        
        if 'error' not in result:
            print(f"Prediction: {result['prediction_label']}")
            print(f"Confidence: {result['confidence']:.4f}")
            print(f"Risk Level: {result['risk_level']}")
            print(f"Risk Score: {result['risk_score']}/100")
            print(f"Probabilities: Malignant={result['probabilities']['malignant']:.4f}, "
                  f"Benign={result['probabilities']['benign']:.4f}")
            print(f"Interpretation: {result['interpretation']}")
            print("Recommendations:")
            for rec in result['recommendations']:
                print(f"  - {rec}")
        else:
            print(f"Error: {result['error']}")
        
        print("\n" + "=" * 50)
        
        # Get feature importance
        print("Top 10 Most Important Features:")
        print("-" * 30)
        importance = predictor.get_feature_importance(top_n=10)
        if 'top_features' in importance:
            for i, (feature, imp) in enumerate(importance['top_features'], 1):
                print(f"{i:2d}. {feature:<35} {imp:.4f}")
        else:
            print(f"Error: {importance.get('error', 'Unknown error')}")
        
    except Exception as e:
        print(f"Error: {str(e)}")

# Alias for backward compatibility with Flask app
BreastCancerPredictor = EnhancedBreastCancerPredictor

if __name__ == "__main__":
    main()