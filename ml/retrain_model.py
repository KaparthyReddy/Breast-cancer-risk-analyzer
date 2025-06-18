#!/usr/bin/env python3
"""
Memory-efficient breast cancer model retraining script
Handles large datasets without memory errors
"""
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import os
import gc  # Garbage collection for memory management

def check_memory_usage():
    """Check current memory usage"""
    import psutil
    process = psutil.Process()
    mem_info = process.memory_info()
    print(f"💾 Memory usage: {mem_info.rss / 1024 / 1024:.1f} MB")

def retrain_model():
    """Memory-efficient model retraining"""
    print("🔄 MEMORY-EFFICIENT MODEL RETRAINING")
    print("=" * 45)
    
    try:
        check_memory_usage()
        
        # Load dataset with memory optimization
        print("📂 Loading dataset...")
        
        # First, peek at the file to understand its structure
        sample_df = pd.read_csv('data/breast_cancer_dataset.csv', nrows=5)
        print(f"✓ Sample data shape: {sample_df.shape}")
        print(f"✓ Columns: {list(sample_df.columns)}")
        
        # Load full dataset with optimized dtypes
        print("📊 Loading full dataset with memory optimization...")
        df = pd.read_csv('data/breast_cancer_dataset.csv', 
                        dtype={col: 'float32' for col in sample_df.select_dtypes(include=[np.number]).columns})
        
        print(f"✓ Full dataset loaded: {df.shape}")
        check_memory_usage()
        
        # Basic data info
        print(f"\n🔍 DATASET OVERVIEW:")
        print(f"   Rows: {df.shape[0]:,}")
        print(f"   Columns: {df.shape[1]}")
        print(f"   Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
        
        # Check for missing values efficiently
        missing_count = df.isnull().sum().sum()
        print(f"   Missing values: {missing_count:,}")
        
        if missing_count > 0:
            print("🧹 Handling missing values...")
            # Handle missing values column by column to save memory
            for col in df.columns:
                if df[col].isnull().sum() > 0:
                    if df[col].dtype in ['float32', 'float64']:
                        df[col].fillna(df[col].median(), inplace=True)
                    else:
                        df[col].fillna(df[col].mode()[0], inplace=True)
        
        # Prepare target variable
        if 'diagnosis' in df.columns:
            print("🎯 Preparing target variable...")
            target_col = 'diagnosis'
            
            # Handle diagnosis conversion efficiently
            if df[target_col].dtype == 'object':
                print(f"   Converting diagnosis to numeric...")
                unique_vals = df[target_col].unique()
                print(f"   Unique values: {unique_vals}")
                
                # Create mapping
                if 'M' in unique_vals or 'B' in unique_vals:
                    df[target_col] = df[target_col].map({'M': 1, 'B': 0})
                elif 'Malignant' in unique_vals or 'Benign' in unique_vals:
                    df[target_col] = df[target_col].map({'Malignant': 1, 'Benign': 0})
            
            # Remove rows with missing target
            before_count = len(df)
            df = df.dropna(subset=[target_col])
            after_count = len(df)
            if before_count != after_count:
                print(f"   Removed {before_count - after_count} rows with missing target")
            
            # Separate features and target
            y = df[target_col].astype('int8')  # Use minimal memory for binary target
            X = df.drop([target_col], axis=1)
            
        else:
            # Assume last column is target
            print("🎯 Using last column as target...")
            y = df.iloc[:, -1].astype('int8')
            X = df.iloc[:, :-1]
        
        # Clean up original dataframe from memory
        del df
        gc.collect()
        check_memory_usage()
        
        print(f"\n✓ Final dataset prepared:")
        print(f"   Features: {X.shape}")
        print(f"   Target distribution: {np.bincount(y)}")
        print(f"   Feature names: {list(X.columns)}")
        
        # Convert features to float32 to save memory
        X = X.astype('float32')
        
        # Split data
        print("\n🔄 Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"   Training set: {X_train.shape}")
        print(f"   Test set: {X_test.shape}")
        check_memory_usage()
        
        # Scale features efficiently
        print("\n⚖️  Scaling features...")
        scaler = StandardScaler()
        
        # Fit scaler on training data
        scaler.fit(X_train)
        
        # Transform data
        X_train_scaled = scaler.transform(X_train).astype('float32')
        X_test_scaled = scaler.transform(X_test).astype('float32')
        
        # Clean up unscaled data
        del X_train, X_test
        gc.collect()
        check_memory_usage()
        
        # Train model with memory-efficient settings
        print("\n🤖 Training RandomForestClassifier...")
        model = RandomForestClassifier(
            n_estimators=50,  # Reduced for memory efficiency
            max_depth=8,      # Reduced depth
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=1,         # Single thread to control memory
            max_samples=0.8   # Use subset of data for each tree
        )
        
        model.fit(X_train_scaled, y_train)
        check_memory_usage()
        
        # Evaluate model
        print("\n📊 Evaluating model...")
        train_score = model.score(X_train_scaled, y_train)
        test_score = model.score(X_test_scaled, y_test)
        
        print(f"✓ Training accuracy: {train_score:.4f}")
        print(f"✓ Test accuracy: {test_score:.4f}")
        
        # Predictions and detailed metrics
        y_pred = model.predict(X_test_scaled)
        print(f"\n📈 CLASSIFICATION REPORT:")
        print(classification_report(y_test, y_pred, target_names=['Benign', 'Malignant']))
        
        # Feature importance (top 10 only to save memory)
        print(f"\n🔍 TOP 10 IMPORTANT FEATURES:")
        feature_names = list(X.columns)
        importances = model.feature_importances_
        
        # Get top 10 features
        top_indices = np.argsort(importances)[-10:][::-1]
        for i, idx in enumerate(top_indices):
            print(f"   {i+1:2d}. {feature_names[idx]:<20} {importances[idx]:.4f}")
        
        # Save model efficiently
        print(f"\n💾 Saving model...")
        os.makedirs('ml', exist_ok=True)
        
        # Prepare model data with minimal memory footprint
        model_data = {
            'model': model,
            'scaler': scaler,
            'feature_names': feature_names,
            'n_features': len(feature_names),
            'training_accuracy': float(train_score),
            'test_accuracy': float(test_score)
        }
        
        joblib.dump(model_data, 'ml/model.pkl', compress=3)  # Compress to save disk space
        
        print(f"✅ MODEL SAVED SUCCESSFULLY!")
        print(f"   📁 Location: ml/model.pkl")
        print(f"   🔢 Features: {len(feature_names)}")
        print(f"   📊 Test accuracy: {test_score:.4f}")
        print(f"   💾 Compressed for efficient storage")
        
        check_memory_usage()
        return True
        
    except MemoryError as e:
        print(f"❌ MEMORY ERROR:")
        print(f"   Your dataset is too large for available RAM")
        print(f"   Suggestions:")
        print(f"   1. Use a smaller sample of your data")
        print(f"   2. Increase system RAM")
        print(f"   3. Use data chunking techniques")
        return False
        
    except Exception as e:
        print(f"❌ ERROR OCCURRED:")
        print(f"   {type(e).__name__}: {e}")
        return False

def create_sample_dataset():
    """Create a smaller sample dataset for testing"""
    print("🔄 Creating sample dataset for testing...")
    try:
        # Read just the first 1000 rows
        df_sample = pd.read_csv('data/breast_cancer_dataset.csv', nrows=1000)
        df_sample.to_csv('data/breast_cancer_sample.csv', index=False)
        print(f"✓ Sample dataset created: data/breast_cancer_sample.csv")
        print(f"   Shape: {df_sample.shape}")
        return True
    except Exception as e:
        print(f"❌ Could not create sample: {e}")
        return False

if __name__ == "__main__":
    # Check available memory first
    try:
        import psutil
        mem = psutil.virtual_memory()
        print(f"💾 SYSTEM MEMORY:")
        print(f"   Total: {mem.total / 1024 / 1024 / 1024:.1f} GB")
        print(f"   Available: {mem.available / 1024 / 1024 / 1024:.1f} GB")
        print(f"   Used: {mem.percent:.1f}%")
        
        if mem.available < 1024 * 1024 * 1024:  # Less than 1GB available
            print(f"⚠️  WARNING: Low memory available!")
            print(f"   Creating sample dataset first...")
            if create_sample_dataset():
                print(f"   Try training on sample first: modify script to use sample file")
        
    except ImportError:
        print("💾 Install psutil for memory monitoring: pip install psutil")
    
    print(f"\n" + "="*50)
    success = retrain_model()
    
    if success:
        print(f"\n🎉 SUCCESS! Model retrained with memory optimization!")
        print(f"📋 Next steps:")
        print(f"   1. Test: python tests/test_prediction.py")
        print(f"   2. Monitor memory usage during inference")
    else:
        print(f"\n❌ Training failed due to memory constraints")
        print(f"💡 Try reducing your dataset size or using more RAM")