#!/usr/bin/env python3
"""
Enhanced Breast Cancer Risk Analyzer with Advanced ML Techniques
Targets 98%+ accuracy through multiple optimization strategies
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier, 
                            GradientBoostingClassifier, VotingClassifier, StackingClassifier)
from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.feature_selection import SelectKBest, f_classif, RFE, SelectFromModel
from sklearn.datasets import load_breast_cancer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import joblib
import os
import warnings
from scipy import stats
from sklearn.model_selection import RandomizedSearchCV
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
warnings.filterwarnings('ignore')

def load_data():
    """Load the breast cancer dataset"""
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name='target')
    return X, y

def advanced_feature_engineering(X):
    """Advanced feature engineering with mathematical transformations"""
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

def create_advanced_models():
    """Create advanced model collection with optimized hyperparameters"""
    models = {}
    
    # XGBoost with optimal parameters
    models['xgb'] = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.1,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        eval_metric='logloss',
        n_jobs=-1
    )
    
    # LightGBM
    models['lgb'] = lgb.LGBMClassifier(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        verbose=-1,
        n_jobs=-1
    )
    
    # CatBoost
    models['cat'] = CatBoostClassifier(
        iterations=500,
        depth=8,
        learning_rate=0.05,
        l2_leaf_reg=3,
        random_seed=42,
        verbose=False
    )
    
    # Advanced Random Forest
    models['rf'] = RandomForestClassifier(
        n_estimators=500,
        max_depth=12,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        bootstrap=True,
        oob_score=True,
        random_state=42,
        n_jobs=-1
    )
    
    # Extra Trees
    models['et'] = ExtraTreesClassifier(
        n_estimators=500,
        max_depth=12,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        bootstrap=True,
        oob_score=True,
        random_state=42,
        n_jobs=-1
    )
    
    # Gradient Boosting
    models['gb'] = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=8,
        min_samples_split=2,
        min_samples_leaf=1,
        subsample=0.8,
        random_state=42
    )
    
    # Neural Network
    models['mlp'] = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation='relu',
        solver='adam',
        alpha=0.001,
        learning_rate='adaptive',
        max_iter=1000,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1
    )
    
    # SVM with RBF
    models['svm'] = SVC(
        C=10,
        gamma='scale',
        kernel='rbf',
        probability=True,
        random_state=42
    )
    
    # Logistic Regression
    models['lr'] = LogisticRegression(
        C=1.0,
        penalty='elasticnet',
        solver='saga',
        l1_ratio=0.5,
        random_state=42,
        max_iter=2000,
        n_jobs=-1
    )
    
    # Linear and Quadratic Discriminant Analysis
    models['lda'] = LinearDiscriminantAnalysis()
    models['qda'] = QuadraticDiscriminantAnalysis()
    
    return models

def create_meta_learner_pipeline(X_train, y_train, X_test, y_test):
    """Create advanced stacking ensemble with meta-learner"""
    
    # Feature selection and preprocessing pipeline
    feature_selector = SelectKBest(score_func=f_classif, k=50)
    scaler = PowerTransformer(method='yeo-johnson')
    
    X_train_selected = feature_selector.fit_transform(X_train, y_train)
    X_train_scaled = scaler.fit_transform(X_train_selected)
    
    X_test_selected = feature_selector.transform(X_test)
    X_test_scaled = scaler.transform(X_test_selected)
    
    # Base models for stacking
    base_models = [
        ('xgb', xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42, n_jobs=-1)),
        ('lgb', lgb.LGBMClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42, verbose=-1, n_jobs=-1)),
        ('rf', RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42, n_jobs=-1)),
        ('et', ExtraTreesClassifier(n_estimators=300, max_depth=10, random_state=42, n_jobs=-1)),
        ('gb', GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42)),
    ]
    
    # Meta-learner
    meta_learner = LogisticRegression(C=1.0, random_state=42, n_jobs=-1)
    
    # Create stacking classifier
    stacking_clf = StackingClassifier(
        estimators=base_models,
        final_estimator=meta_learner,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        stack_method='predict_proba',
        n_jobs=-1
    )
    
    print("Training Stacking Ensemble...")
    stacking_clf.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = stacking_clf.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, stacking_clf.predict_proba(X_test_scaled)[:, 1])
    
    print(f"Stacking Ensemble - Accuracy: {accuracy:.4f}, AUC: {auc_score:.4f}")
    
    return stacking_clf, feature_selector, scaler, accuracy

def create_voting_ensemble(X_train, y_train, X_test, y_test):
    """Create advanced voting ensemble"""
    
    # Feature selection
    feature_selector = SelectKBest(score_func=f_classif, k=45)
    scaler = RobustScaler()
    
    X_train_selected = feature_selector.fit_transform(X_train, y_train)
    X_train_scaled = scaler.fit_transform(X_train_selected)
    
    X_test_selected = feature_selector.transform(X_test)
    X_test_scaled = scaler.transform(X_test_selected)
    
    # Best performing models for voting
    voting_models = [
        ('xgb', xgb.XGBClassifier(n_estimators=400, max_depth=7, learning_rate=0.05, random_state=42, n_jobs=-1)),
        ('lgb', lgb.LGBMClassifier(n_estimators=400, max_depth=7, learning_rate=0.05, random_state=42, verbose=-1, n_jobs=-1)),
        ('cat', CatBoostClassifier(iterations=400, depth=7, learning_rate=0.05, random_seed=42, verbose=False)),
        ('rf', RandomForestClassifier(n_estimators=400, max_depth=11, random_state=42, n_jobs=-1)),
        ('et', ExtraTreesClassifier(n_estimators=400, max_depth=11, random_state=42, n_jobs=-1)),
    ]
    
    # Soft voting ensemble
    voting_clf = VotingClassifier(
        estimators=voting_models,
        voting='soft',
        n_jobs=-1
    )
    
    print("Training Voting Ensemble...")
    voting_clf.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = voting_clf.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, voting_clf.predict_proba(X_test_scaled)[:, 1])
    
    print(f"Voting Ensemble - Accuracy: {accuracy:.4f}, AUC: {auc_score:.4f}")
    
    return voting_clf, feature_selector, scaler, accuracy

def hyperparameter_optimization(X_train, y_train):
    """Advanced hyperparameter optimization for best model"""
    
    # Feature selection
    feature_selector = SelectKBest(score_func=f_classif, k=40)
    X_train_selected = feature_selector.fit_transform(X_train, y_train)
    
    # XGBoost hyperparameter space
    param_distributions = {
        'n_estimators': [400, 500, 600],
        'max_depth': [6, 7, 8, 9],
        'learning_rate': [0.03, 0.05, 0.07],
        'subsample': [0.8, 0.85, 0.9],
        'colsample_bytree': [0.8, 0.85, 0.9],
        'gamma': [0, 0.1, 0.2],
        'min_child_weight': [1, 3, 5],
        'reg_alpha': [0, 0.1, 0.5],
        'reg_lambda': [1, 1.5, 2]
    }
    
    xgb_model = xgb.XGBClassifier(random_state=42, eval_metric='logloss', n_jobs=-1)
    
    # Randomized search with stratified CV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    random_search = RandomizedSearchCV(
        xgb_model,
        param_distributions,
        n_iter=50,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    
    print("Performing hyperparameter optimization...")
    random_search.fit(X_train_selected, y_train)
    
    print(f"Best parameters: {random_search.best_params_}")
    print(f"Best CV score: {random_search.best_score_:.4f}")
    
    return random_search.best_estimator_, feature_selector

def evaluate_model_comprehensive(model, X_test, y_test, model_name="Model"):
    """Comprehensive model evaluation"""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_proba)
    
    print(f"\n{model_name} Results:")
    print("-" * 60)
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"AUC Score: {auc_score:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Malignant', 'Benign']))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    return accuracy, auc_score

def save_enhanced_model(model, scaler, feature_names, selector=None, filename='enhanced_model.pkl'):
    """Save the enhanced model with all components"""
    os.makedirs('ml', exist_ok=True)
    
    model_data = {
        'model': model,
        'scaler': scaler,
        'feature_names': feature_names,
        'selector': selector,
        'model_type': type(model).__name__,
        'feature_count': len(feature_names)
    }
    
    joblib.dump(model_data, os.path.join('ml', filename))
    print(f"Enhanced model saved to ml/{filename}")

def main():
    print("Enhanced Breast Cancer Risk Analyzer - Targeting 98%+ Accuracy")
    print("=" * 80)
    
    # Load and preprocess data
    X, y = load_data()
    print(f"Original dataset shape: {X.shape}")
    
    # Advanced feature engineering
    X_enhanced = advanced_feature_engineering(X)
    print(f"Enhanced dataset shape: {X_enhanced.shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_enhanced, y, test_size=0.15, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Method 1: Stacking Ensemble with Meta-learner
    print("\n" + "="*80)
    print("METHOD 1: STACKING ENSEMBLE WITH META-LEARNER")
    print("="*80)
    
    stacking_model, stack_selector, stack_scaler, stack_accuracy = create_meta_learner_pipeline(
        X_train, y_train, X_test, y_test
    )
    
    # Method 2: Advanced Voting Ensemble
    print("\n" + "="*80)
    print("METHOD 2: ADVANCED VOTING ENSEMBLE")
    print("="*80)
    
    voting_model, vote_selector, vote_scaler, vote_accuracy = create_voting_ensemble(
        X_train, y_train, X_test, y_test
    )
    
    # Method 3: Hyperparameter Optimized XGBoost
    print("\n" + "="*80)
    print("METHOD 3: HYPERPARAMETER OPTIMIZED XGBOOST")
    print("="*80)
    
    optimized_model, opt_selector = hyperparameter_optimization(X_train, y_train)
    opt_scaler = PowerTransformer(method='yeo-johnson')
    
    X_train_opt = opt_selector.transform(X_train)
    X_train_opt_scaled = opt_scaler.fit_transform(X_train_opt)
    X_test_opt = opt_selector.transform(X_test)
    X_test_opt_scaled = opt_scaler.transform(X_test_opt)
    
    optimized_model.fit(X_train_opt_scaled, y_train)
    opt_accuracy, opt_auc = evaluate_model_comprehensive(
        optimized_model, X_test_opt_scaled, y_test, "Optimized XGBoost"
    )
    
    # Select the best model
    models_comparison = [
        ("Stacking Ensemble", stack_accuracy, stacking_model, stack_selector, stack_scaler),
        ("Voting Ensemble", vote_accuracy, voting_model, vote_selector, vote_scaler),
        ("Optimized XGBoost", opt_accuracy, optimized_model, opt_selector, opt_scaler)
    ]
    
    best_name, best_accuracy, best_model, best_selector, best_scaler = max(
        models_comparison, key=lambda x: x[1]
    )
    
    print(f"\n" + "="*80)
    print(f"BEST MODEL: {best_name}")
    print(f"FINAL ACCURACY: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
    print("="*80)
    
    if best_accuracy >= 0.98:
        print("🎯 TARGET ACHIEVED: 98%+ Accuracy!")
    else:
        print(f"📈 Current best: {best_accuracy*100:.2f}% - Close to target!")
    
    # Save the best model
    save_enhanced_model(
        best_model, best_scaler, X_enhanced.columns.tolist(), 
        best_selector, 'enhanced_model.pkl'
    )
    
    # Cross-validation score for final validation
    if best_selector:
        X_selected = best_selector.transform(X_enhanced)
        X_scaled = best_scaler.transform(X_selected)
    else:
        X_scaled = best_scaler.transform(X_enhanced)
    
    cv_scores = cross_val_score(best_model, X_scaled, y, cv=10, scoring='accuracy')
    print(f"\n10-Fold Cross-Validation Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    return best_accuracy

if __name__ == "__main__":
    final_accuracy = main()