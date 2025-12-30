"""
================================================================================
Random Forest Driver Analysis
================================================================================

Objective B: Drivers of Habitat Loss Analysis - Greater Kafue Ecosystem

This script performs Random Forest classification to identify key drivers:
    - Feature selection with Recursive Feature Elimination (RFE)
    - Spatial cross-validation for robust model evaluation
    - Variable importance analysis (Gini and Permutation)
    - Critical threshold identification using Youden's J statistic
    - Partial dependence analysis for driver relationships

Author: Gift Mulenga
Institution: Copperbelt University, Zambia
Research: MSc Thesis - Tropical Ecology

Requirements:
    - scikit-learn
    - pandas
    - numpy
    - matplotlib
    - seaborn
    - joblib

================================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold, GroupKFold
from sklearn.feature_selection import RFE
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, auc, confusion_matrix, cohen_kappa_score
)
from sklearn.tree import export_text, plot_tree
import joblib
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality plot defaults
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10


# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Configuration for Random Forest analysis."""
    
    # Directory paths - UPDATE THESE
    BASE_DIR = r"D:/Publication/Habitat Loss in the Greater Kafue Ecosystem/outputs/objective_b"
    INPUT_DIR = os.path.join(BASE_DIR, "samples")
    OUTPUT_DIR = os.path.join(BASE_DIR, "results")
    TABLES_DIR = os.path.join(BASE_DIR, "tables")
    FIGURES_DIR = os.path.join(BASE_DIR, "figures")
    
    # Model parameters
    N_ESTIMATORS = 500
    MAX_DEPTH = None
    MIN_SAMPLES_SPLIT = 10
    MIN_SAMPLES_LEAF = 5
    MAX_FEATURES = 'sqrt'
    CLASS_WEIGHT = 'balanced'
    RANDOM_STATE = 42
    N_JOBS = -1
    
    # Cross-validation
    CV_FOLDS = 10
    USE_SPATIAL_CV = True
    
    # Feature selection
    USE_RFE = True
    RFE_N_FEATURES = 10
    
    # Variables to exclude
    VARS_TO_EXCLUDE = ['years_protected']
    
    # Variable categories
    VARIABLE_CATEGORIES = {
        'dist_roads': 'Proximity', 'dist_settlements': 'Proximity',
        'dist_rivers': 'Proximity', 'dist_knp': 'Proximity',
        'pop_density': 'Socio-economic', 'pop_change': 'Socio-economic',
        'pct_cultivated': 'Socio-economic', 'protection_status': 'Conservation',
        'elevation': 'Topographic', 'slope': 'Topographic',
        'aspect': 'Topographic', 'twi': 'Topographic',
        'mean_rainfall': 'Climatic', 'mean_temp': 'Climatic',
    }
    
    # Display names
    VARIABLE_DISPLAY_NAMES = {
        'dist_roads': 'Distance to Roads',
        'dist_settlements': 'Distance to Settlements',
        'dist_rivers': 'Distance to Rivers',
        'dist_knp': 'Distance to KNP',
        'pop_density': 'Population Density',
        'pop_change': 'Population Change',
        'pct_cultivated': 'Percent Cultivated',
        'protection_status': 'Protection Status',
        'elevation': 'Elevation',
        'slope': 'Slope',
        'aspect': 'Aspect',
        'twi': 'Topographic Wetness Index',
        'mean_rainfall': 'Mean Rainfall',
        'mean_temp': 'Mean Temperature',
    }
    
    # Category colors
    CATEGORY_COLORS = {
        'Socio-economic': '#E74C3C',
        'Climatic': '#3498DB',
        'Proximity': '#2ECC71',
        'Topographic': '#9B59B6',
        'Conservation': '#F39C12',
    }


# =============================================================================
# DATA LOADING
# =============================================================================

def create_output_directories():
    """Create output directories."""
    for d in [Config.OUTPUT_DIR, Config.TABLES_DIR, Config.FIGURES_DIR]:
        os.makedirs(d, exist_ok=True)
    print("✓ Output directories ready")


def load_data():
    """Load training and test data."""
    print("\n--- Loading Data ---")
    
    train_df = pd.read_csv(os.path.join(Config.INPUT_DIR, 'train_data.csv'))
    test_df = pd.read_csv(os.path.join(Config.INPUT_DIR, 'test_data.csv'))
    
    print(f"  Train samples: {len(train_df)}")
    print(f"  Test samples: {len(test_df)}")
    
    # Identify features
    exclude_cols = ['x', 'y', 'habitat_loss', 'row', 'col', 'spatial_block'] + Config.VARS_TO_EXCLUDE
    features = [c for c in train_df.columns if c not in exclude_cols]
    
    print(f"  Features: {len(features)}")
    print(f"  Excluded: {Config.VARS_TO_EXCLUDE}")
    
    return train_df, test_df, features


# =============================================================================
# FEATURE SELECTION
# =============================================================================

def perform_rfe(X_train, y_train, features, n_features_to_select=10):
    """
    Perform Recursive Feature Elimination.
    
    Args:
        X_train: Training features
        y_train: Training labels
        features: Feature names
        n_features_to_select: Number of features to select
        
    Returns:
        tuple: (selected features, ranking DataFrame)
    """
    print("\n--- Recursive Feature Elimination ---")
    
    rf_rfe = RandomForestClassifier(
        n_estimators=100, max_depth=10,
        random_state=Config.RANDOM_STATE, n_jobs=Config.N_JOBS
    )
    
    rfe = RFE(rf_rfe, n_features_to_select=n_features_to_select, step=1)
    rfe.fit(X_train, y_train)
    
    ranking_df = pd.DataFrame({
        'Variable': features,
        'RFE_Rank': rfe.ranking_,
        'Selected': rfe.support_
    }).sort_values('RFE_Rank')
    
    selected_features = ranking_df[ranking_df['Selected']]['Variable'].tolist()
    
    print(f"  Selected {len(selected_features)} features:")
    for f in selected_features:
        print(f"    - {f}")
    
    return selected_features, ranking_df


# =============================================================================
# MODEL TRAINING
# =============================================================================

def train_model(X_train, y_train):
    """
    Train Random Forest classifier.
    
    Args:
        X_train: Training features
        y_train: Training labels
        
    Returns:
        Trained model
    """
    print("\n--- Training Random Forest ---")
    
    model = RandomForestClassifier(
        n_estimators=Config.N_ESTIMATORS,
        max_depth=Config.MAX_DEPTH,
        min_samples_split=Config.MIN_SAMPLES_SPLIT,
        min_samples_leaf=Config.MIN_SAMPLES_LEAF,
        max_features=Config.MAX_FEATURES,
        class_weight=Config.CLASS_WEIGHT,
        random_state=Config.RANDOM_STATE,
        n_jobs=Config.N_JOBS,
        oob_score=True
    )
    
    model.fit(X_train, y_train)
    
    print(f"  OOB Score: {model.oob_score_:.4f}")
    
    return model


# =============================================================================
# CROSS-VALIDATION
# =============================================================================

def spatial_cross_validation(model, X, y, groups, n_folds):
    """
    Perform spatial (grouped) cross-validation.
    
    Args:
        model: Classifier
        X: Features
        y: Labels
        groups: Spatial block assignments
        n_folds: Number of folds
        
    Returns:
        dict: Cross-validation scores
    """
    print("\n--- Spatial Cross-Validation ---")
    
    gkf = GroupKFold(n_splits=min(n_folds, len(np.unique(groups))))
    
    scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    cv_results = cross_validate(model, X, y, groups=groups, cv=gkf, scoring=scoring)
    
    scores = {
        'accuracy': cv_results['test_accuracy'],
        'precision': cv_results['test_precision'],
        'recall': cv_results['test_recall'],
        'f1': cv_results['test_f1'],
        'roc_auc': cv_results['test_roc_auc']
    }
    
    print("\n  CV Results (mean ± std):")
    for metric, values in scores.items():
        print(f"    {metric}: {np.mean(values):.4f} ± {np.std(values):.4f}")
    
    return scores


def standard_cross_validation(model, X, y, n_folds):
    """Perform standard stratified cross-validation."""
    print("\n--- Standard Cross-Validation ---")
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=Config.RANDOM_STATE)
    
    scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    cv_results = cross_validate(model, X, y, cv=skf, scoring=scoring)
    
    scores = {
        'accuracy': cv_results['test_accuracy'],
        'precision': cv_results['test_precision'],
        'recall': cv_results['test_recall'],
        'f1': cv_results['test_f1'],
        'roc_auc': cv_results['test_roc_auc']
    }
    
    print("\n  CV Results (mean ± std):")
    for metric, values in scores.items():
        print(f"    {metric}: {np.mean(values):.4f} ± {np.std(values):.4f}")
    
    return scores


# =============================================================================
# MODEL EVALUATION
# =============================================================================

def evaluate_model(model, X_test, y_test, features):
    """
    Evaluate model on test set.
    
    Args:
        model: Trained classifier
        X_test: Test features
        y_test: Test labels
        features: Feature names
        
    Returns:
        tuple: (metrics dict, confusion matrix)
    """
    print("\n--- Model Evaluation ---")
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba),
        'kappa': cohen_kappa_score(y_test, y_pred)
    }
    
    cm = confusion_matrix(y_test, y_pred)
    
    print("\n  Test Set Performance:")
    for metric, value in metrics.items():
        print(f"    {metric}: {value:.4f}")
    
    print(f"\n  Confusion Matrix:")
    print(f"    TN: {cm[0, 0]:,}  FP: {cm[0, 1]:,}")
    print(f"    FN: {cm[1, 0]:,}  TP: {cm[1, 1]:,}")
    
    # Save metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(os.path.join(Config.TABLES_DIR, 'Table_Model_Performance.csv'), index=False)
    print("\n  ✓ Performance metrics saved")
    
    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {metrics["roc_auc"]:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(Config.FIGURES_DIR, 'Figure_ROC_Curve.png'), dpi=300)
    plt.close()
    print("  ✓ ROC curve saved")
    
    return metrics, cm


# =============================================================================
# VARIABLE IMPORTANCE
# =============================================================================

def analyze_importance(model, X_train, y_train, X_test, y_test, features):
    """
    Analyze variable importance using multiple methods.
    
    Args:
        model: Trained classifier
        X_train, y_train: Training data
        X_test, y_test: Test data
        features: Feature names
        
    Returns:
        tuple: (importance DataFrame, category summary)
    """
    print("\n--- Variable Importance Analysis ---")
    
    # Gini importance
    gini_importance = model.feature_importances_
    
    # Permutation importance
    perm_result = permutation_importance(
        model, X_test, y_test,
        n_repeats=30, random_state=Config.RANDOM_STATE, n_jobs=Config.N_JOBS
    )
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        'Variable': features,
        'Gini_Importance': gini_importance,
        'Permutation_Importance': perm_result.importances_mean,
        'Permutation_Std': perm_result.importances_std
    })
    
    # Add ranks
    importance_df['Gini_Rank'] = importance_df['Gini_Importance'].rank(ascending=False).astype(int)
    importance_df['Permutation_Rank'] = importance_df['Permutation_Importance'].rank(ascending=False).astype(int)
    
    # Add categories
    importance_df['Category'] = importance_df['Variable'].map(Config.VARIABLE_CATEGORIES)
    importance_df['Display_Name'] = importance_df['Variable'].map(Config.VARIABLE_DISPLAY_NAMES)
    
    # Sort by Gini importance
    importance_df = importance_df.sort_values('Gini_Importance', ascending=False)
    
    print("\n  Top 10 Variables (Gini):")
    for i, row in importance_df.head(10).iterrows():
        print(f"    {row['Gini_Rank']:2d}. {row['Display_Name']}: {row['Gini_Importance']:.4f}")
    
    # Save
    importance_df.to_csv(os.path.join(Config.TABLES_DIR, 'Table_Variable_Importance.csv'), index=False)
    print("\n  ✓ Variable importance table saved")
    
    # Category summary
    category_summary = importance_df.groupby('Category').agg({
        'Gini_Importance': 'sum',
        'Permutation_Importance': 'sum'
    }).reset_index()
    
    category_summary['Gini_Pct'] = category_summary['Gini_Importance'] / category_summary['Gini_Importance'].sum() * 100
    category_summary = category_summary.sort_values('Gini_Importance', ascending=False)
    
    category_summary.to_csv(os.path.join(Config.TABLES_DIR, 'Table_Category_Importance.csv'), index=False)
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Bar chart
    colors = [Config.CATEGORY_COLORS.get(c, 'gray') for c in importance_df['Category']]
    axes[0].barh(importance_df['Display_Name'].head(10)[::-1], 
                 importance_df['Gini_Importance'].head(10)[::-1],
                 color=colors[:10][::-1])
    axes[0].set_xlabel('Gini Importance')
    axes[0].set_title('Top 10 Variables')
    
    # Category pie
    axes[1].pie(category_summary['Gini_Pct'], labels=category_summary['Category'],
                autopct='%1.1f%%', colors=[Config.CATEGORY_COLORS.get(c, 'gray') 
                                           for c in category_summary['Category']])
    axes[1].set_title('Importance by Category')
    
    plt.tight_layout()
    plt.savefig(os.path.join(Config.FIGURES_DIR, 'Figure_Variable_Importance.png'), dpi=300)
    plt.close()
    print("  ✓ Variable importance figure saved")
    
    return importance_df, category_summary


# =============================================================================
# THRESHOLD ANALYSIS
# =============================================================================

def analyze_thresholds(model, X_train, y_train, features):
    """
    Identify critical thresholds using Youden's J statistic.
    
    Args:
        model: Trained classifier
        X_train: Training features
        y_train: Training labels
        features: Feature names
        
    Returns:
        DataFrame: Threshold analysis results
    """
    print("\n--- Critical Threshold Analysis ---")
    
    results = []
    
    for i, feature in enumerate(features):
        # Get feature values
        X_single = X_train[:, i].reshape(-1, 1)
        
        # Simple logistic-style analysis
        loss_values = X_single[y_train == 1].flatten()
        no_loss_values = X_single[y_train == 0].flatten()
        
        # Find optimal threshold using ROC
        thresholds = np.percentile(X_single, np.arange(5, 96, 5))
        
        best_j = 0
        best_thresh = np.median(X_single)
        best_sens = 0
        best_spec = 0
        
        for thresh in thresholds:
            pred = (X_single.flatten() <= thresh).astype(int)
            tp = np.sum((pred == 1) & (y_train == 1))
            tn = np.sum((pred == 0) & (y_train == 0))
            fp = np.sum((pred == 1) & (y_train == 0))
            fn = np.sum((pred == 0) & (y_train == 1))
            
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0
            j = sens + spec - 1
            
            if j > best_j:
                best_j = j
                best_thresh = thresh
                best_sens = sens
                best_spec = spec
        
        results.append({
            'Variable': feature,
            'Display_Name': Config.VARIABLE_DISPLAY_NAMES.get(feature, feature),
            'Optimal_Threshold': best_thresh,
            'Youden_J': best_j,
            'Sensitivity': best_sens,
            'Specificity': best_spec,
            'Mean_Loss': np.mean(loss_values),
            'Mean_NoLoss': np.mean(no_loss_values)
        })
    
    threshold_df = pd.DataFrame(results).sort_values('Youden_J', ascending=False)
    threshold_df.to_csv(os.path.join(Config.TABLES_DIR, 'Table_Critical_Thresholds.csv'), index=False)
    
    print("\n  Top 5 Discriminating Variables:")
    for i, row in threshold_df.head(5).iterrows():
        print(f"    {row['Display_Name']}: threshold = {row['Optimal_Threshold']:.1f}, J = {row['Youden_J']:.3f}")
    
    print("\n  ✓ Threshold analysis saved")
    
    return threshold_df


# =============================================================================
# PARTIAL DEPENDENCE
# =============================================================================

def create_pdp(model, X_train, features):
    """
    Create partial dependence plots for top variables.
    
    Args:
        model: Trained classifier
        X_train: Training features
        features: Feature names
    """
    print("\n--- Creating Partial Dependence Plots ---")
    
    # Get top 6 features
    importance = model.feature_importances_
    top_idx = np.argsort(importance)[-6:][::-1]
    
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    
    for idx, (ax, feat_idx) in enumerate(zip(axes.flatten(), top_idx)):
        PartialDependenceDisplay.from_estimator(
            model, X_train, [feat_idx],
            ax=ax, grid_resolution=50
        )
        display_name = Config.VARIABLE_DISPLAY_NAMES.get(features[feat_idx], features[feat_idx])
        ax.set_title(display_name, fontsize=11)
        ax.set_xlabel('')
    
    plt.suptitle('Partial Dependence Plots', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(Config.FIGURES_DIR, 'Figure_Partial_Dependence.png'), dpi=300)
    plt.close()
    
    print("  ✓ Partial dependence plots saved")


# =============================================================================
# MODEL PERSISTENCE
# =============================================================================

def save_model_and_metadata(model, features, metrics, cv_scores):
    """Save trained model and metadata."""
    print("\n--- Saving Model and Metadata ---")
    
    # Save model
    model_path = os.path.join(Config.OUTPUT_DIR, 'rf_model.joblib')
    joblib.dump(model, model_path)
    
    # Save metadata
    metadata = {
        'model_params': {
            'n_estimators': Config.N_ESTIMATORS,
            'max_depth': Config.MAX_DEPTH,
            'min_samples_split': Config.MIN_SAMPLES_SPLIT,
            'min_samples_leaf': Config.MIN_SAMPLES_LEAF,
            'max_features': Config.MAX_FEATURES
        },
        'features': features,
        'test_metrics': metrics,
        'cv_metrics': {k: {'mean': float(np.mean(v)), 'std': float(np.std(v))} 
                       for k, v in cv_scores.items()},
        'timestamp': datetime.now().isoformat()
    }
    
    with open(os.path.join(Config.OUTPUT_DIR, 'model_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("  ✓ Model and metadata saved")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function."""
    print("="*60)
    print("RANDOM FOREST DRIVER ANALYSIS")
    print("Greater Kafue Ecosystem - Habitat Loss Drivers")
    print("="*60)
    
    print(f"\nConfiguration:")
    print(f"  - Spatial CV: {Config.USE_SPATIAL_CV}")
    print(f"  - RFE: {Config.USE_RFE}")
    print(f"  - Variables excluded: {Config.VARS_TO_EXCLUDE}")
    
    create_output_directories()
    
    # Load data
    train_df, test_df, features = load_data()
    
    # Prepare arrays
    X_train = train_df[features].values
    y_train = train_df['habitat_loss'].values
    X_test = test_df[features].values
    y_test = test_df['habitat_loss'].values
    
    # Feature selection
    if Config.USE_RFE:
        selected_features, rfe_ranking = perform_rfe(X_train, y_train, features, Config.RFE_N_FEATURES)
        rfe_ranking.to_csv(os.path.join(Config.TABLES_DIR, 'Table_RFE_Ranking.csv'), index=False)
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Cross-validation
    if Config.USE_SPATIAL_CV and 'spatial_block' in train_df.columns:
        groups = train_df['spatial_block'].values
        cv_scores = spatial_cross_validation(model, X_train, y_train, groups, Config.CV_FOLDS)
    else:
        cv_scores = standard_cross_validation(model, X_train, y_train, Config.CV_FOLDS)
    
    # Evaluation
    metrics, cm = evaluate_model(model, X_test, y_test, features)
    
    # Variable importance
    importance_df, category_summary = analyze_importance(
        model, X_train, y_train, X_test, y_test, features
    )
    
    # Threshold analysis
    threshold_df = analyze_thresholds(model, X_train, y_train, features)
    
    # Partial dependence
    create_pdp(model, X_train, features)
    
    # Save model
    save_model_and_metadata(model, features, metrics, cv_scores)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"\nOutputs saved to:")
    print(f"  Tables: {Config.TABLES_DIR}")
    print(f"  Figures: {Config.FIGURES_DIR}")
    print(f"  Model: {Config.OUTPUT_DIR}")
    
    return {
        'model': model,
        'features': features,
        'metrics': metrics,
        'importance': importance_df,
        'cv_scores': cv_scores,
        'thresholds': threshold_df
    }


if __name__ == "__main__":
    results = main()
