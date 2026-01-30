"""
Project: Biometric Weight Estimation
Module: Model Serialization for Deployment
Description: Trains and serializes the optimized XGBoost model for production deployment
             in the Streamlit web application. Uses best hyperparameters from optimization.

Technical Specifications:
    - Input: Processed facial features CSV (66,724 samples)
    - Output: Serialized XGBoost model (pickle format)
    - Hyperparameters: Optimized via RandomizedSearchCV
    - Model Size: ~72 KB (highly portable)

Author: Surya Yalavarthi
Institution: University of Cincinnati
Date: January 2026
"""

import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

# Configuration
RANDOM_SEED = 42
FEATURES_PATH = '/Users/suryayalavarthi/Downloads/Bodyweight Predication/idoc_weight_estimation/facial_features_ratios_V2.csv'
LABELS_PATH = '/Users/suryayalavarthi/Downloads/Bodyweight Predication/idoc_weight_estimation/data/raw_images/archive/labels_utf8.csv'
MODEL_OUTPUT_PATH = '/Users/suryayalavarthi/Downloads/Bodyweight Predication/xgboost_weight_model.pkl'

print("="*70)
print("SAVING OPTIMIZED XGBOOST MODEL FOR DEPLOYMENT")
print("="*70)

# Load and merge data
print("\n1. Loading data...")
features_df = pd.read_csv(FEATURES_PATH)
labels_df = pd.read_csv(LABELS_PATH)

# Rename ID column
labels_df = labels_df.rename(columns={'ID': 'filename'})

# Merge
merged_df = pd.merge(features_df, labels_df, on='filename', how='inner')
print(f"   ✓ Loaded {len(merged_df):,} samples")

# Parse weight
merged_df['weight'] = merged_df['Weight'].str.extract(r'(\d+)').astype(float)
merged_df = merged_df.drop(columns=['Weight'])

# Clean data
facial_cols = ['left_eyebrow_ratio', 'right_eyebrow_ratio', 'left_eye_ratio', 
               'right_eye_ratio', 'nose_width_ratio', 'nose_length_ratio',
               'outer_lip_ratio', 'inner_lip_ratio', 'face_height_ratio']
merged_df = merged_df.dropna(subset=['weight'] + facial_cols)
print(f"   ✓ Clean dataset: {len(merged_df):,} samples")

# Prepare features
print("\n2. Preparing features...")
X = merged_df[facial_cols]
y = merged_df['weight']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED
)
print(f"   ✓ Training set: {len(X_train):,} samples")

# Train model with optimized hyperparameters
print("\n3. Training model with optimized hyperparameters...")
best_model = XGBRegressor(
    n_estimators=40,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.9,
    min_child_weight=5,
    gamma=0.1,
    reg_alpha=0.01,
    reg_lambda=1,
    random_state=RANDOM_SEED,
    n_jobs=-1,
    verbosity=0
)

best_model.fit(X_train, y_train)
print("   ✓ Model trained successfully")

# Save model
print(f"\n4. Saving model to: {MODEL_OUTPUT_PATH}")
with open(MODEL_OUTPUT_PATH, 'wb') as f:
    pickle.dump(best_model, f)
print("   ✓ Model saved successfully")

# Verify model
print("\n5. Verifying saved model...")
with open(MODEL_OUTPUT_PATH, 'rb') as f:
    loaded_model = pickle.load(f)

test_prediction = loaded_model.predict(X_test[:5])
print(f"   ✓ Test predictions: {test_prediction}")

print("\n" + "="*70)
print("✅ MODEL READY FOR STREAMLIT DEPLOYMENT!")
print("="*70)
print(f"\nModel file: {MODEL_OUTPUT_PATH}")
print("Next step: Run 'streamlit run streamlit_app.py'")
print("="*70)
