#!/usr/bin/env python3
"""
Optimized Model Generator for Breast Cancer Prediction
Memory-efficient version targeting 98%+ accuracy with stable performance
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier, 
                            GradientBoostingClassifier, VotingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.datasets import load_breast_cancer
from sklearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgb
import joblib
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

class OptimizedModelGenerator:
    """
    Memory-efficient model generator with optimized feature engineering.
    """
    
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_pipeline = None
        self.best_accuracy = 0.0
        
    def load_data(self):
        """Load the breast cancer dataset."""
        print("Loading breast cancer dataset...")
        data = load_breast_cancer()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.Series(data.target, name='target')
        print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
        return X, y
    
    def remove_outliers(self, X, y, contamination=0.03):
        """Remove outliers using Isolation Forest."""
        from sklearn.ensemble import IsolationForest
        
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        outlier_labels = iso_forest.fit_predict(X)
        
        mask = outlier_labels == 1
        X_clean = X[mask]
        y_clean = y[mask]
        
        print(f"Removed {np.sum(~mask)} outliers ({np.sum(~mask)/len(X)*100:.2f}%)")
        return X_clean, y_clean
    
    def optimized_feature_engineering(self, X):
        """Create essential engineered features without memory overhead."""
        print("Applying optimized feature engineering...")
        X_enhanced = X.copy()
        
        # High-impact ratio features
        X_enhanced['area_perimeter_ratio'] = X['mean area'] / (X['mean perimeter'] + 1e-8)
        X_enhanced['compactness_concavity_ratio'] = X['mean compactness'] / (X['mean concavity'] + 1e-8)
        X_enhanced['worst_concave_area_product'] = X['worst concave points'] * X['worst area']
        
        # Severity indicators (worst vs mean)
        key_features = ['radius', 'texture', 'perimeter', 'area', 'smoothness', 
                       'compactness', 'concavity', 'concave points', 'symmetry']
        
        for feature in key_features:
            mean_col = f'mean {feature}'
            worst_col = f'worst {feature}'
            if mean_col in X.columns and worst_col in X.columns:
                X_enhanced[f'{feature}_severity'] = X[worst_col] / (X[mean_col] + 1e-8)
                X_enhanced[f'{feature}_range'] = X[worst_col] - X[mean_col]
        
        # Top performing polynomial features (based on analysis)
        top_features = ['worst concave points', 'worst perimeter', 'worst radius', 
                       'mean concave points', 'worst area']
        
        for feature in top_features:
            if feature in X.columns:
                X_enhanced[f'{feature}_squared'] = X[feature] ** 2
                X_enhanced[f'{feature}_log'] = np.log1p(X[feature])
        
        # Critical interactions
        X_enhanced['concave_points_area'] = X['worst concave points'] * X['worst area']
        X_enhanced['radius_perimeter_interaction'] = X['mean radius'] * X['mean perimeter']
        X_enhanced['compactness_concavity_worst'] = X['worst compactness'] * X['worst concavity']
        
        # Geometric estimates
        X_enhanced['volume_estimate'] = (X['mean area'] ** 1.5) / (X['mean perimeter'] + 1e-8)
        X_enhanced['irregularity_score'] = X['worst fractal dimension'] * X['worst symmetry']
        
        # Clean infinite and NaN values
        X_enhanced = X_enhanced.replace([np.inf, -np.inf], np.nan)
        X_enhanced = X_enhanced.fillna(X_enhanced.median())
        
        print(f"Feature engineering completed: {X_enhanced.shape[1]} features created")
        return X_enhanced
    
    def create_optimized_models(self):
        """Create optimized models with memory-efficient parameters."""
        models = {}
        
        # XGBoost - Optimized for performance and stability
        models['xgb'] = xgb.XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.1,
            min_child_weight=1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            eval_metric='logloss',
            n_jobs=4  # Limit parallelism
        )
        
        # LightGBM - Memory efficient
        models['lgb'] = lgb.LGBMClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=10,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            verbose=-1,
            n_jobs=4,
            num_leaves=31
        )
        
        # Random Forest - Optimized
        models['rf'] = RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_split=3,
            min_samples_leaf=2,
            max_features='sqrt',
            bootstrap=True,
            random_state=42,
            n_jobs=4,
            class_weight='balanced'
        )
        
        # Extra Trees
        models['et'] = ExtraTreesClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_split=3,
            min_samples_leaf=2,
            max_features='sqrt',
            bootstrap=True,
            random_state=42,
            n_jobs=4,
            class_weight='balanced'
        )
        
        # Gradient Boosting
        models['gb'] = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            min_samples_split=3,
            min_samples_leaf=2,
            subsample=0.8,
            random_state=42,
            max_features='sqrt'
        )
        
        # Support Vector Machine
        models['svm'] = SVC(
            C=10.0,
            gamma='scale',
            kernel='rbf',
            probability=True,
            random_state=42,
            class_weight='balanced'
        )
        
        # Logistic Regression
        models['lr'] = LogisticRegression(
            C=1.0,
            penalty='l2',
            solver='liblinear',
            random_state=42,
            max_iter=1000,
            class_weight='balanced'
        )
        
        return models
    
    def create_memory_efficient_ensemble(self, trained_models, X_train, y_train):
        """Create a memory-efficient voting ensemble."""
        # Select top performing models
        ensemble_models = []
        for name, model in trained_models.items():
            if name in ['xgb', 'lgb', 'rf', 'et', 'gb']:  # Skip SVM and LR for ensemble
                ensemble_models.append((name, model))
        
        voting_ensemble = VotingClassifier(
            estimators=ensemble_models,
            voting='soft',
            n_jobs=2  # Limit parallelism
        )
        
        return voting_ensemble
    
    def evaluate_models_with_pipelines(self, X_train, X_test, y_train, y_test):
        """Evaluate models using pipelines for better memory management."""
        print("\nEvaluating models with optimized pipelines...")
        print("=" * 60)
        
        models = self.create_optimized_models()
        results = {}
        trained_models = {}
        
        # Feature selection and scaling options
        selector = SelectKBest(score_func=f_classif, k=50)
        scaler = RobustScaler()
        
        print("Individual Model Performance:")
        print("-" * 40)
        
        for name, model in models.items():
            try:
                # Create pipeline
                pipeline = Pipeline([
                    ('selector', selector),
                    ('scaler', scaler),
                    ('model', model)
                ])
                
                # Fit pipeline
                pipeline.fit(X_train, y_train)
                
                # Predict
                y_pred = pipeline.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                
                results[name] = accuracy
                trained_models[name] = pipeline
                print(f"{name.upper():<8}: {accuracy:.4f} ({accuracy*100:.2f}%)")
                
            except Exception as e:
                print(f"{name.upper():<8}: Error - {str(e)[:50]}")
                continue
        
        # Create ensemble with preprocessed data
        print("\nCreating ensemble model...")
        try:
            # Get the best individual models for ensemble
            best_individual_models = {}
            X_train_processed = selector.fit_transform(X_train, y_train)
            X_train_scaled = scaler.fit_transform(X_train_processed)
            X_test_processed = selector.transform(X_test)
            X_test_scaled = scaler.transform(X_test_processed)
            
            # Train models on processed data
            base_models = self.create_optimized_models()
            for name in ['xgb', 'lgb', 'rf', 'et', 'gb']:
                if name in base_models:
                    model = base_models[name]
                    model.fit(X_train_scaled, y_train)
                    best_individual_models[name] = model
            
            # Create voting ensemble
            ensemble_models = [(name, model) for name, model in best_individual_models.items()]
            voting_ensemble = VotingClassifier(
                estimators=ensemble_models,
                voting='soft',
                n_jobs=2
            )
            
            voting_ensemble.fit(X_train_scaled, y_train)
            y_pred_ensemble = voting_ensemble.predict(X_test_scaled)
            ensemble_accuracy = accuracy_score(y_test, y_pred_ensemble)
            
            # Create ensemble pipeline
            ensemble_pipeline = Pipeline([
                ('selector', selector),
                ('scaler', scaler),
                ('ensemble', voting_ensemble)
            ])
            
            results['ensemble'] = ensemble_accuracy
            trained_models['ensemble'] = ensemble_pipeline
            print(f"ENSEMBLE : {ensemble_accuracy:.4f} ({ensemble_accuracy*100:.2f}%)")
            
        except Exception as e:
            print(f"ENSEMBLE : Error - {str(e)[:50]}")
        
        # Select best model
        best_model_name = max(results, key=results.get)
        best_accuracy = results[best_model_name]
        best_pipeline = trained_models[best_model_name]
        
        print(f"\nBest Model: {best_model_name.upper()}")
        print(f"Best Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
        
        self.best_model = best_model_name
        self.best_pipeline = best_pipeline
        self.best_accuracy = best_accuracy
        
        return best_pipeline, best_accuracy
    
    def cross_validate_best_model(self, X, y):
        """Perform cross-validation on the best model."""
        print(f"\nPerforming 10-fold cross-validation on {self.best_model.upper()}...")
        
        # Cross-validation with stratified folds
        cv_scores = cross_val_score(
            self.best_pipeline, X, y, 
            cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=42), 
            scoring='accuracy', n_jobs=2
        )
        
        print(f"Cross-validation scores: {[f'{score:.4f}' for score in cv_scores]}")
        print(f"Mean CV accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        print(f"CV accuracy range: {cv_scores.min():.4f} - {cv_scores.max():.4f}")
        
        return cv_scores
    
    def detailed_evaluation(self, X_test, y_test):
        """Provide detailed evaluation of the best model."""
        print(f"\nDetailed Evaluation of Best Model ({self.best_model.upper()}):")
        print("=" * 50)
        
        # Predictions
        y_pred = self.best_pipeline.predict(X_test)
        y_pred_proba = self.best_pipeline.predict_proba(X_test)
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_pred_proba[:, 1])
        
        print(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"AUC Score: {auc_score:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Malignant', 'Benign']))
        
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(f"[[{cm[0,0]}, {cm[0,1]}],")
        print(f" [{cm[1,0]}, {cm[1,1]}]]")
        
        return accuracy, auc_score
    
    def save_model(self, filepath="optimized_breast_cancer_model.pkl"):
        """Save the best model pipeline."""
        model_package = {
            'pipeline': self.best_pipeline,
            'model_name': self.best_model,
            'accuracy': self.best_accuracy,
            'timestamp': datetime.now().isoformat()
        }
        
        joblib.dump(model_package, filepath)
        print(f"\nModel saved to {filepath}")
        print(f"Model type: {self.best_model.upper()}")
        print(f"Final accuracy: {self.best_accuracy:.4f} ({self.best_accuracy*100:.2f}%)")
    
    def generate_model(self):
        """Main method to generate the optimized model."""
        print("🚀 Optimized Breast Cancer Prediction Model Generator")
        print("=" * 60)
        
        # Load data
        X, y = self.load_data()
        
        # Remove outliers
        X_clean, y_clean = self.remove_outliers(X, y)
        
        # Optimized feature engineering
        X_enhanced = self.optimized_feature_engineering(X_clean)
        
        # Split data with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X_enhanced, y_clean, test_size=0.2, random_state=42, stratify=y_clean
        )
        
        print(f"Training set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")
        
        # Evaluate models
        best_pipeline, best_accuracy = self.evaluate_models_with_pipelines(
            X_train, X_test, y_train, y_test
        )
        
        # Cross-validation
        cv_scores = self.cross_validate_best_model(X_enhanced, y_clean)
        
        # Detailed evaluation
        final_accuracy, auc_score = self.detailed_evaluation(X_test, y_test)
        
        # Save the model
        self.save_model()
        
        print(f"\n🎯 FINAL RESULTS:")
        print(f"Best Model: {self.best_model.upper()}")
        print(f"Test Accuracy: {final_accuracy:.4f} ({final_accuracy*100:.2f}%)")
        print(f"Cross-validation Mean: {cv_scores.mean():.4f} ({cv_scores.mean()*100:.2f}%)")
        print(f"AUC Score: {auc_score:.4f}")
        
        if final_accuracy >= 0.98:
            print("🏆 TARGET ACHIEVED: 98%+ accuracy reached!")
        elif final_accuracy >= 0.97:
            print("🎉 EXCELLENT: 97%+ accuracy achieved!")
        else:
            print(f"📈 Strong Performance: {final_accuracy*100:.2f}% accuracy achieved")
        
        return self.best_pipeline

def main():
    """Main execution function."""
    generator = OptimizedModelGenerator()
    model = generator.generate_model()
    return generator

if __name__ == "__main__":
    generator = main()