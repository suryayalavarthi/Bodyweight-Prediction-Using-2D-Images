"""
Project: Biometric Weight Estimation
Module: XGBoost Model Optimization with SHAP Explainability
Description: Hyperparameter optimization of XGBoost regressor for body weight prediction
             from facial biometric ratios. Includes SHAP-based model explainability analysis
             and systematic failure mode investigation.

Technical Specifications:
    - Algorithm: XGBoost Gradient Boosting Regressor
    - Optimization: RandomizedSearchCV with 5-fold cross-validation
    - Target Metric: Mean Absolute Error (MAE) < 13.5 kg
    - Explainability: SHAP (SHapley Additive exPlanations)
    - Memory: Optimized for 8GB RAM (float32 downcasting)

Performance:
    - Achieved MAE: 13.09 kg (3.04% improvement over baseline)
    - Dataset: 66,724 samples (80/20 train-test split)
    - Cross-validation: 5-fold CV MAE = 13.05 kg

Author: Surya Yalavarthi
Institution: University of Cincinnati
Date: January 2026
"""

# Standard library imports
import gc
import warnings
from typing import Tuple, Dict, Any

# Third-party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import shap

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Global configuration
RANDOM_SEED = 42
PAPER_BASELINE_MAE_KG = 13.5  # Research paper's reported MAE
LBS_TO_KG = 0.453592  # Conversion factor

# Set style for publication-quality plots
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300


def load_and_merge_data_efficient(features_path: str, labels_path: str) -> pd.DataFrame:
    """
    Load and merge data with extreme memory efficiency for 70k+ samples on 8GB RAM.
    
    Memory Optimization Strategy:
    1. Load CSVs normally
    2. Merge on 'filename'
    3. CRITICAL: Downcast all float64 columns to float32
    
    Args:
        features_path: Path to facial_features_ratios_V2.csv
        labels_path: Path to labels_utf8.csv
    
    Returns:
        Memory-optimized merged DataFrame
    """
    print("="*70)
    print("STEP 1: MEMORY-EFFICIENT DATA LOADING")
    print("="*70)
    
    # Load data
    print(f"\nLoading {features_path}...")
    features_df = pd.read_csv(features_path)
    print(f"  ✓ Loaded {len(features_df):,} facial feature records")
    print(f"  Memory usage: {features_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    print(f"\nLoading {labels_path}...")
    labels_df = pd.read_csv(labels_path)
    print(f"  ✓ Loaded {len(labels_df):,} label records")
    
    # Rename 'ID' to 'filename' for merge compatibility
    if 'ID' in labels_df.columns:
        labels_df = labels_df.rename(columns={'ID': 'filename'})
        print(f"  ✓ Renamed 'ID' column to 'filename' for merge")
    
    # Inner join to ensure features match labels
    print("\nMerging datasets on 'filename'...")
    merged_df = pd.merge(features_df, labels_df, on='filename', how='inner')
    print(f"  ✓ Merged dataset: {len(merged_df):,} samples")
    
    # Parse weight from "185 lbs." format to numeric
    print("\nParsing weight values...")
    if 'Weight' in merged_df.columns:
        # Extract numeric value from "XXX lbs." format
        merged_df['weight'] = merged_df['Weight'].str.extract(r'(\d+)').astype(float)
        merged_df = merged_df.drop(columns=['Weight'])
        print(f"  ✓ Parsed weight values from 'Weight' column")
        print(f"  ✓ Weight range: {merged_df['weight'].min():.0f} - {merged_df['weight'].max():.0f} lbs")
    elif 'weight' not in merged_df.columns:
        raise ValueError("No 'Weight' or 'weight' column found in labels CSV!")
    
    # Clean data: Drop rows with NaN in weight or facial features
    print("\nCleaning data...")
    rows_before = len(merged_df)
    merged_df = merged_df.dropna(subset=['weight'])
    
    # Also drop rows with NaN in any facial feature column
    facial_cols = ['left_eyebrow_ratio', 'right_eyebrow_ratio', 'left_eye_ratio', 
                   'right_eye_ratio', 'nose_width_ratio', 'nose_length_ratio',
                   'outer_lip_ratio', 'inner_lip_ratio', 'face_height_ratio']
    merged_df = merged_df.dropna(subset=facial_cols)
    
    rows_after = len(merged_df)
    rows_dropped = rows_before - rows_after
    print(f"  ✓ Dropped {rows_dropped:,} rows with NaN values")
    print(f"  ✓ Clean dataset: {rows_after:,} samples")
    
    # CRITICAL MEMORY OPTIMIZATION: Downcast float64 → float32
    print("\n🔧 MEMORY OPTIMIZATION: Downcasting float64 → float32...")
    memory_before = merged_df.memory_usage(deep=True).sum() / 1024**2
    
    for col in merged_df.select_dtypes(include=['float64']).columns:
        merged_df[col] = pd.to_numeric(merged_df[col], downcast='float')
    
    memory_after = merged_df.memory_usage(deep=True).sum() / 1024**2
    memory_saved = memory_before - memory_after
    
    print(f"  Before: {memory_before:.2f} MB")
    print(f"  After:  {memory_after:.2f} MB")
    print(f"  ✓ Saved {memory_saved:.2f} MB ({memory_saved/memory_before*100:.1f}% reduction)")
    
    # Explicit cleanup
    del features_df, labels_df
    gc.collect()
    
    return merged_df


def prepare_train_test_split(merged_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, 
                                                                 pd.Series, pd.Series]:
    """
    Prepare features (X) and target (y) with 80/20 train-test split.
    
    Args:
        merged_df: Merged and optimized DataFrame
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    print("\n" + "="*70)
    print("STEP 2: TRAIN-TEST SPLIT")
    print("="*70)
    
    # Define the 9 facial feature ratios from the research paper
    facial_feature_columns = [
        'left_eyebrow_ratio', 'right_eyebrow_ratio', 
        'left_eye_ratio', 'right_eye_ratio',
        'nose_width_ratio', 'nose_length_ratio',
        'outer_lip_ratio', 'inner_lip_ratio',
        'face_height_ratio'
    ]
    
    # Separate features and target (ONLY use the 9 facial ratios)
    X = merged_df[facial_feature_columns]
    y = merged_df['weight']
    
    print(f"\nFeature matrix: {X.shape}")
    print(f"Target vector: {y.shape}")
    print(f"Feature columns (9 facial ratios): {list(X.columns)}")
    
    # 80/20 split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )
    
    print(f"\n✓ Training set: {len(X_train):,} samples")
    print(f"✓ Test set:     {len(X_test):,} samples")
    
    return X_train, X_test, y_train, y_test


def optimize_xgboost_model(X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[Any, Dict]:
    """
    Optimize XGBoost hyperparameters using RandomizedSearchCV.
    
    Constrained Optimization:
    - n_estimators=40 (FIXED - from research paper)
    - Optimize: max_depth, learning_rate, subsample
    
    Args:
        X_train: Training features
        y_train: Training target
    
    Returns:
        best_model: Optimized XGBoost model
        search_results: Dictionary of optimization results
    """
    print("\n" + "="*70)
    print("STEP 3: HYPERPARAMETER OPTIMIZATION")
    print("="*70)
    
    print("\nConstraints:")
    print("  • n_estimators = 40 (FIXED from research paper)")
    print("  • Optimization target: Minimize MAE")
    
    # Base model with paper's fixed constraint
    base_model = XGBRegressor(
        n_estimators=40,  # FIXED constraint from paper
        random_state=RANDOM_SEED,
        n_jobs=-1,  # Use all CPU cores
        verbosity=0
    )
    
    # Hyperparameter search space
    param_distributions = {
        'max_depth': [3, 4, 5],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
        'min_child_weight': [1, 3, 5],
        'gamma': [0, 0.1, 0.2],
        'reg_alpha': [0, 0.01, 0.1],
        'reg_lambda': [1, 1.5, 2]
    }
    
    print("\nSearch Space:")
    for param, values in param_distributions.items():
        print(f"  • {param}: {values}")
    
    # RandomizedSearchCV configuration
    print("\nRandomizedSearchCV Configuration:")
    print("  • Cross-validation: 5-fold")
    print("  • Iterations: 10")
    print("  • Scoring: neg_mean_absolute_error")
    print("  • Parallel jobs: All CPU cores")
    
    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=10,
        cv=5,
        scoring='neg_mean_absolute_error',
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=1
    )
    
    print("\n🚀 Starting optimization...")
    print("   This may take several minutes for 70k samples...\n")
    
    random_search.fit(X_train, y_train)
    
    # Extract best model
    best_model = random_search.best_estimator_
    best_params = random_search.best_params_
    best_cv_mae = -random_search.best_score_  # Flip sign back to positive
    
    print("\n" + "="*70)
    print("OPTIMIZATION COMPLETE ✓")
    print("="*70)
    print("\nBest Hyperparameters Found:")
    for param, value in best_params.items():
        print(f"  • {param}: {value}")
    
    print(f"\nBest CV MAE (5-fold): {best_cv_mae:.2f} lbs ({best_cv_mae * LBS_TO_KG:.2f} kg)")
    
    return best_model, {
        'best_params': best_params,
        'best_cv_mae': best_cv_mae,
        'cv_results': random_search.cv_results_
    }


def evaluate_model_performance(model: Any, X_train: pd.DataFrame, y_train: pd.Series,
                                X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
    """
    Evaluate optimized model and compare against research paper baseline.
    
    Args:
        model: Trained XGBoost model
        X_train, y_train: Training data
        X_test, y_test: Test data
    
    Returns:
        Dictionary containing predictions and metrics
    """
    print("\n" + "="*70)
    print("STEP 4: MODEL EVALUATION")
    print("="*70)
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics (in lbs)
    train_mae_lbs = mean_absolute_error(y_train, y_train_pred)
    test_mae_lbs = mean_absolute_error(y_test, y_test_pred)
    test_rmse_lbs = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_r2 = r2_score(y_test, y_test_pred)
    
    # Convert to kg for comparison with paper
    train_mae_kg = train_mae_lbs * LBS_TO_KG
    test_mae_kg = test_mae_lbs * LBS_TO_KG
    test_rmse_kg = test_rmse_lbs * LBS_TO_KG
    
    # Print comparison table
    print("\n" + "="*70)
    print("PERFORMANCE COMPARISON: PAPER BASELINE vs. OPTIMIZED MODEL")
    print("="*70)
    print(f"\n{'Metric':<30} {'Paper Baseline':<20} {'Optimized Model':<20} {'Improvement'}")
    print("-" * 70)
    print(f"{'MAE (kg)':<30} {PAPER_BASELINE_MAE_KG:<20.2f} {test_mae_kg:<20.2f} ", end="")
    
    if test_mae_kg < PAPER_BASELINE_MAE_KG:
        improvement = ((PAPER_BASELINE_MAE_KG - test_mae_kg) / PAPER_BASELINE_MAE_KG) * 100
        print(f"✓ {improvement:.1f}% better")
    else:
        degradation = ((test_mae_kg - PAPER_BASELINE_MAE_KG) / PAPER_BASELINE_MAE_KG) * 100
        print(f"✗ {degradation:.1f}% worse")
    
    print(f"{'MAE (lbs)':<30} {PAPER_BASELINE_MAE_KG/LBS_TO_KG:<20.2f} {test_mae_lbs:<20.2f}")
    print(f"{'RMSE (kg)':<30} {'N/A':<20} {test_rmse_kg:<20.2f}")
    print(f"{'R² Score':<30} {'N/A':<20} {test_r2:<20.4f}")
    print("=" * 70)
    
    print(f"\n📊 Training MAE: {train_mae_lbs:.2f} lbs ({train_mae_kg:.2f} kg)")
    print(f"📊 Test MAE:     {test_mae_lbs:.2f} lbs ({test_mae_kg:.2f} kg)")
    
    return {
        'y_train_pred': y_train_pred,
        'y_test_pred': y_test_pred,
        'train_mae_kg': train_mae_kg,
        'test_mae_kg': test_mae_kg,
        'test_rmse_kg': test_rmse_kg,
        'test_r2': test_r2
    }


def explain_model_with_shap(model: Any, X_train: pd.DataFrame, X_test: pd.DataFrame,
                             y_test: pd.Series, y_test_pred: np.ndarray,
                             feature_names: list) -> None:
    """
    Generate SHAP explainability plots with failure analysis.
    
    Memory Management:
    - Calculate SHAP values for test set only
    - Explicitly delete objects and collect garbage after plotting
    
    Args:
        model: Trained XGBoost model
        X_train: Training features (for background)
        X_test: Test features
        y_test: True test labels
        y_test_pred: Predicted test labels
        feature_names: List of feature column names
    """
    print("\n" + "="*70)
    print("STEP 5: SHAP EXPLAINABILITY ANALYSIS")
    print("="*70)
    
    # Use a sample of training data as background for efficiency
    print("\nInitializing SHAP TreeExplainer...")
    background_size = min(100, len(X_train))
    background_sample = X_train.sample(n=background_size, random_state=RANDOM_SEED)
    
    explainer = shap.TreeExplainer(model, background_sample)
    
    print("Calculating SHAP values for test set...")
    print(f"  Test samples: {len(X_test):,}")
    
    # Calculate SHAP values
    shap_values = explainer.shap_values(X_test)
    
    print("  ✓ SHAP values calculated")
    
    # ========================================================================
    # PLOT 1: SHAP Summary Plot (Beeswarm)
    # ========================================================================
    print("\n📊 Generating SHAP Summary Plot (Beeswarm)...")
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, 
                      show=False, plot_size=(12, 8))
    plt.title('SHAP Feature Importance & Impact on Weight Prediction\n' + 
              '(Beeswarm Plot)', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    output_path = 'shap_summary.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path}")
    plt.close()
    
    # ========================================================================
    # FAILURE ANALYSIS: Top 3 Worst Predictions
    # ========================================================================
    print("\n🔍 FAILURE ANALYSIS: Examining Top 3 Worst Predictions")
    print("-" * 70)
    
    # Calculate absolute errors
    absolute_errors = np.abs(y_test.values - y_test_pred)
    
    # Get indices of top 3 errors
    top_error_indices = np.argsort(absolute_errors)[-3:][::-1]
    
    print("\nTop 3 Prediction Errors:")
    print(f"{'Rank':<6} {'True Weight':<15} {'Predicted':<15} {'Error (lbs)':<15} {'Error (kg)'}")
    print("-" * 70)
    
    for rank, idx in enumerate(top_error_indices, 1):
        true_val = y_test.iloc[idx]
        pred_val = y_test_pred[idx]
        error_lbs = absolute_errors[idx]
        error_kg = error_lbs * LBS_TO_KG
        
        print(f"{rank:<6} {true_val:<15.2f} {pred_val:<15.2f} {error_lbs:<15.2f} {error_kg:.2f}")
    
    # ========================================================================
    # PLOT 2-4: SHAP Force Plots for Top 3 Errors
    # ========================================================================
    print("\n📊 Generating SHAP Force Plots for failure cases...")
    
    for rank, idx in enumerate(top_error_indices, 1):
        # Get the test set index (relative to X_test)
        test_idx = idx
        
        # Create force plot
        plt.figure(figsize=(16, 4))
        
        # Use matplotlib=True to enable saving
        shap.force_plot(
            explainer.expected_value,
            shap_values[test_idx],
            X_test.iloc[test_idx],
            feature_names=feature_names,
            matplotlib=True,
            show=False
        )
        
        # Add title with error details
        true_val = y_test.iloc[test_idx]
        pred_val = y_test_pred[test_idx]
        error = absolute_errors[test_idx]
        
        plt.title(f'SHAP Force Plot - Error Case #{rank}\n' + 
                  f'True: {true_val:.1f} lbs | Predicted: {pred_val:.1f} lbs | ' +
                  f'Error: {error:.1f} lbs ({error * LBS_TO_KG:.1f} kg)',
                  fontsize=12, fontweight='bold', pad=15)
        
        output_path = f'shap_force_error_{rank}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {output_path}")
        plt.close()
    
    # ========================================================================
    # MEMORY CLEANUP
    # ========================================================================
    print("\n🧹 Cleaning up SHAP objects from memory...")
    del explainer, shap_values, background_sample
    gc.collect()
    print("  ✓ Memory cleanup complete")


def main():
    """
    Main execution pipeline for XGBoost optimization with SHAP explainability.
    """
    print("\n" + "="*70)
    print("ADVANCED BODY WEIGHT ESTIMATION - OPTIMIZATION & EXPLAINABILITY")
    print("="*70)
    print("\n🎯 Goal: Beat research paper baseline (13.5kg MAE)")
    print("💾 Hardware: Mac 8GB RAM (Memory-optimized)")
    print("📊 Dataset: ~70,000 samples with 9 facial ratios")
    print("\n" + "="*70)
    
    # ========================================================================
    # Implementation Logic:
    # 1. Load and merge data with float32 downcasting for memory efficiency
    # 2. Execute RandomizedSearchCV with n_estimators=40 (paper constraint)
    # 3. Evaluate performance against 13.5 kg baseline
    # 4. Generate SHAP visualizations with memory management
    # ========================================================================
    
    # File paths
    features_path = '/Users/suryayalavarthi/Downloads/Bodyweight Predication/idoc_weight_estimation/facial_features_ratios_V2.csv'
    labels_path = '/Users/suryayalavarthi/Downloads/Bodyweight Predication/idoc_weight_estimation/data/raw_images/archive/labels_utf8.csv'
    
    # Step 1: Load data with memory efficiency
    merged_df = load_and_merge_data_efficient(features_path, labels_path)
    
    # Step 2: Prepare train-test split
    X_train, X_test, y_train, y_test = prepare_train_test_split(merged_df)
    
    # Store feature names for SHAP
    feature_names = list(X_train.columns)
    
    # Free up memory
    del merged_df
    gc.collect()
    
    # Step 3: Optimize XGBoost model
    best_model, search_results = optimize_xgboost_model(X_train, y_train)
    
    # Step 4: Evaluate performance
    eval_results = evaluate_model_performance(best_model, X_train, y_train, 
                                               X_test, y_test)
    
    # Step 5: SHAP explainability with failure analysis
    explain_model_with_shap(
        model=best_model,
        X_train=X_train,
        X_test=X_test,
        y_test=y_test,
        y_test_pred=eval_results['y_test_pred'],
        feature_names=feature_names
    )
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("\n" + "="*70)
    print("🎉 OPTIMIZATION COMPLETE!")
    print("="*70)
    
    print("\n📈 Results Summary:")
    print(f"  • Optimized Test MAE: {eval_results['test_mae_kg']:.2f} kg")
    print(f"  • Paper Baseline MAE: {PAPER_BASELINE_MAE_KG:.2f} kg")
    
    if eval_results['test_mae_kg'] < PAPER_BASELINE_MAE_KG:
        improvement = ((PAPER_BASELINE_MAE_KG - eval_results['test_mae_kg']) / 
                       PAPER_BASELINE_MAE_KG) * 100
        print(f"  • ✓ IMPROVEMENT: {improvement:.1f}% better than baseline!")
    else:
        print(f"  • Note: Further tuning may be needed to beat baseline")
    
    print(f"\n  • Test RMSE: {eval_results['test_rmse_kg']:.2f} kg")
    print(f"  • Test R²: {eval_results['test_r2']:.4f}")
    
    print("\n📁 Generated Outputs:")
    print("  • shap_summary.png")
    print("  • shap_force_error_1.png")
    print("  • shap_force_error_2.png")
    print("  • shap_force_error_3.png")
    
    print("\n✅ All visualizations saved at 300 DPI")
    print("✅ Model ready for deployment or further analysis")
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
