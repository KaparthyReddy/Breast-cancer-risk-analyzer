import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.datasets import load_breast_cancer
from sklearn.pipeline import Pipeline
import xgboost as xgb
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load the breast cancer dataset"""
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name='target')
    return X, y

def feature_engineering(X):
    """Create additional features from existing ones"""
    X_enhanced = X.copy()
    
    # Create ratio features (these often work well for medical data)
    X_enhanced['mean_area_to_perimeter_ratio'] = X['mean area'] / X['mean perimeter']
    X_enhanced['mean_compactness_to_concavity_ratio'] = X['mean compactness'] / (X['mean concavity'] + 1e-8)
    X_enhanced['texture_smoothness_interaction'] = X['mean texture'] * X['mean smoothness']
    
    # Create polynomial features for top features
    important_features = ['mean concave points', 'worst perimeter', 'worst concave points', 'mean texture']
    for feature in important_features:
        if feature in X.columns:
            X_enhanced[f'{feature}_squared'] = X[feature] ** 2
    
    return X_enhanced

def preprocess_data(X, y):
    """Enhanced preprocessing with feature engineering"""
    # Feature engineering
    X_enhanced = feature_engineering(X)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X_enhanced, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Use RobustScaler instead of StandardScaler (better for outliers)
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X_enhanced.columns.tolist(), X_test, y_test

def create_optimized_models():
    """Create a collection of optimized models"""
    models = {}
    
    # Optimized Random Forest
    models['rf'] = RandomForestClassifier(
        n_estimators=300,  # Reduced from 500
        max_depth=15,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        bootstrap=True,
        random_state=42,
        n_jobs=1  # Use single job to prevent memory issues
    )
    
    # XGBoost (often performs very well)
    models['xgb'] = xgb.XGBClassifier(
        n_estimators=200,  # Reduced from 300
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss',
        tree_method='hist',  # More memory efficient
        n_jobs=1  # Use single job
    )
    
    # Extra Trees (another ensemble method)
    models['et'] = ExtraTreesClassifier(
        n_estimators=200,  # Reduced from 300
        max_depth=15,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=1  # Use single job
    )
    
    # Logistic Regression with regularization
    models['lr'] = LogisticRegression(
        C=1.0,
        penalty='l2',
        solver='liblinear',
        random_state=42,
        max_iter=1000
    )
    
    # SVM with RBF kernel
    models['svm'] = SVC(
        C=10,
        gamma='scale',
        kernel='rbf',
        probability=True,
        random_state=42
    )
    
    return models

def train_ensemble_model(X_train, y_train, X_test, y_test, feature_names):
    """Train an ensemble of models with feature selection"""
    
    # Feature selection using SelectKBest
    selector = SelectKBest(score_func=f_classif, k=25)  # Select top 25 features
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    
    # Get individual models
    models = create_optimized_models()
    
    # Train individual models and evaluate
    trained_models = {}
    individual_scores = {}
    
    print("Training individual models:")
    print("-" * 50)
    
    for name, model in models.items():
        model.fit(X_train_selected, y_train)
        y_pred = model.predict(X_test_selected)
        accuracy = accuracy_score(y_test, y_pred)
        trained_models[name] = model
        individual_scores[name] = accuracy
        print(f"{name.upper()}: {accuracy:.4f}")
    
    # Create voting ensemble with the best performing models
    # Use soft voting for better performance
    voting_clf = VotingClassifier(
        estimators=[
            ('rf', trained_models['rf']),
            ('xgb', trained_models['xgb']),
            ('et', trained_models['et']),
            ('lr', trained_models['lr']),
            ('svm', trained_models['svm'])
        ],
        voting='soft'  # Use probability averaging
    )
    
    voting_clf.fit(X_train_selected, y_train)
    
    return voting_clf, selector, individual_scores

def hyperparameter_tuning(X_train, y_train):
    """Perform hyperparameter tuning for XGBoost (usually the best performer)"""
    print("Performing hyperparameter tuning...")
    
    # Feature selection
    selector = SelectKBest(score_func=f_classif, k=25)
    X_train_selected = selector.fit_transform(X_train, y_train)
    
    # Reduced XGBoost parameter grid to prevent memory issues
    param_grid = {
        'n_estimators': [200, 400],
        'max_depth': [4, 6],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.8, 0.9]
    }
    
    xgb_model = xgb.XGBClassifier(
        random_state=42, 
        eval_metric='logloss',
        tree_method='hist',  # More memory efficient
        n_jobs=1  # Use single thread to prevent memory issues
    )
    
    # Use StratifiedKFold for better cross-validation
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)  # Reduced folds
    
    grid_search = GridSearchCV(
        xgb_model, 
        param_grid, 
        cv=cv, 
        scoring='accuracy',
        n_jobs=1,  # Use single job to prevent memory issues
        verbose=1
    )
    
    try:
        grid_search.fit(X_train_selected, y_train)
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
        return grid_search.best_estimator_, selector
    except Exception as e:
        print(f"Hyperparameter tuning failed: {e}")
        print("Using default XGBoost parameters...")
        # Fallback to default parameters
        default_xgb = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42,
            eval_metric='logloss',
            tree_method='hist',
            n_jobs=1
        )
        default_xgb.fit(X_train_selected, y_train)
        return default_xgb, selector

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """Evaluate model performance"""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n{model_name} Results:")
    print("-" * 50)
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    return accuracy

def save_model(model, scaler, feature_names, selector=None, filename='model.pkl'):
    """Save the trained model and preprocessing components"""
    os.makedirs('ml', exist_ok=True)
    
    model_data = {
        'model': model,
        'scaler': scaler,
        'feature_names': feature_names,
        'selector': selector
    }
    
    joblib.dump(model_data, os.path.join('ml', filename))
    print(f"Enhanced model saved to ml/{filename}")

def save_test_data(X_test, y_test):
    """Save test data for later evaluation"""
    joblib.dump({'X_test': X_test, 'y_test': y_test}, 'ml/test_data.pkl')
    print("Test data saved to ml/test_data.pkl")

def main():
    print("Training Enhanced Breast Cancer Risk Analyzer")
    print("=" * 60)
    
    # Load and preprocess data
    X, y = load_data()
    X_train, X_test, y_train, y_test, scaler, feature_names, X_test_raw, y_test_raw = preprocess_data(X, y)
    
    print(f"Dataset shape: {X.shape}")
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Features after engineering: {len(feature_names)}")
    
    # Method 1: Ensemble approach
    print("\n" + "="*60)
    print("METHOD 1: ENSEMBLE OF MULTIPLE MODELS")
    print("="*60)
    
    ensemble_model, selector, individual_scores = train_ensemble_model(
        X_train, y_train, X_test, y_test, feature_names
    )
    
    X_test_selected = selector.transform(X_test)
    ensemble_accuracy = evaluate_model(ensemble_model, X_test_selected, y_test, "Ensemble Model")
    
    # Method 2: Hyperparameter tuned XGBoost
    print("\n" + "="*60)
    print("METHOD 2: HYPERPARAMETER TUNED XGBOOST")
    print("="*60)
    
    tuned_model, tuned_selector = hyperparameter_tuning(X_train, y_train)
    X_test_tuned = tuned_selector.transform(X_test)
    tuned_accuracy = evaluate_model(tuned_model, X_test_tuned, y_test, "Tuned XGBoost")
    
    # Choose the best model
    if ensemble_accuracy >= tuned_accuracy:
        best_model = ensemble_model
        best_selector = selector
        best_accuracy = ensemble_accuracy
        model_type = "Ensemble"
    else:
        best_model = tuned_model
        best_selector = tuned_selector
        best_accuracy = tuned_accuracy
        model_type = "Tuned XGBoost"
    
    print(f"\n" + "="*60)
    print(f"BEST MODEL: {model_type}")
    print(f"FINAL ACCURACY: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
    print("="*60)
    
    # Save the best model
    save_model(best_model, scaler, feature_names, best_selector, 'enhanced_model.pkl')
    save_test_data(X_test_raw, y_test_raw)
    
    # Feature importance for interpretability
    if hasattr(best_model, 'feature_importances_'):
        selected_features = np.array(feature_names)[best_selector.get_support()]
        importance_df = pd.DataFrame({
            'feature': selected_features,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop 10 Most Important Features:")
        print("-" * 40)
        for i, (_, row) in enumerate(importance_df.head(10).iterrows()):
            print(f"{i+1:2d}. {row['feature']:<35} {row['importance']:.4f}")
    
    return best_accuracy

if __name__ == "__main__":
    final_accuracy = main()