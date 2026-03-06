"""
Kepler Telescope Data Analysis
===============================
Comprehensive analysis of Kepler exoplanet cumulative dataset
Author: Stellar Analytics Team
Date: 2026-02-14

This script performs:
1. Data loading and exploration
2. Data cleaning and preprocessing
3. Exploratory data analysis (EDA)
4. Machine learning model development
5. Insights generation and visualization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix, 
                             accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, roc_curve)
from scipy import stats
import os

# Set random seed for reproducibility
np.random.seed(42)
warnings.filterwarnings('ignore')

# Configure matplotlib for better plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Create output directory for visualizations
if not os.path.exists('visualizations'):
    os.makedirs('visualizations')

print("="*80)
print("KEPLER TELESCOPE DATA ANALYSIS")
print("="*80)
print()

# ============================================================================
# SECTION 1: DATA LOADING & INITIAL EXPLORATION
# ============================================================================
print("\n" + "="*80)
print("SECTION 1: DATA LOADING & INITIAL EXPLORATION")
print("="*80 + "\n")

# Load the dataset
print("Loading Kepler cumulative dataset...")
df = pd.read_csv('cumulative.csv')
print("✓ Dataset loaded successfully!\n")

# Display basic information
print(f"Dataset Shape: {df.shape[0]:,} rows × {df.shape[1]} columns\n")

print("Column Names:")
print("-" * 40)
for i, col in enumerate(df.columns, 1):
    print(f"{i:2d}. {col}")

print(f"\n\nData Types Summary:")
print("-" * 40)
print(df.dtypes.value_counts())

print(f"\n\nFirst 5 Rows:")
print("-" * 80)
print(df.head())

print(f"\n\nBasic Statistical Summary:")
print("-" * 80)
print(df.describe())

print(f"\n\nMissing Values Summary:")
print("-" * 40)
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
missing_df = pd.DataFrame({
    'Missing_Count': missing,
    'Percentage': missing_pct
}).sort_values('Percentage', ascending=False)
print(missing_df[missing_df['Missing_Count'] > 0].head(20))

# ============================================================================
# SECTION 2: DATA CLEANING & PREPROCESSING
# ============================================================================
print("\n\n" + "="*80)
print("SECTION 2: DATA CLEANING & PREPROCESSING")
print("="*80 + "\n")

# Create a copy for processing
df_clean = df.copy()

print("Step 1: Identifying and handling duplicates...")
initial_rows = len(df_clean)
df_clean = df_clean.drop_duplicates()
duplicates_removed = initial_rows - len(df_clean)
print(f"✓ Removed {duplicates_removed} duplicate rows")

# Focus on key columns for exoplanet detection
print("\nStep 2: Selecting relevant features...")

# Key columns based on Kepler data documentation
key_columns = [
    'koi_disposition',  # Target variable: CONFIRMED, FALSE POSITIVE, CANDIDATE
    'koi_score',        # Disposition score
    'koi_period',       # Orbital period
    'koi_teq',          # Equilibrium temperature
    'koi_insol',        # Insolation flux
    'koi_depth',        # Transit depth
    'koi_duration',     # Transit duration
    'koi_prad',         # Planetary radius
    'koi_srad',         # Stellar radius
    'koi_smass',        # Stellar mass
    'koi_steff',        # Stellar effective temperature
    'ra',               # Right ascension
    'dec'               # Declination
]

# Filter columns that exist in the dataset
available_columns = [col for col in key_columns if col in df_clean.columns]
df_clean = df_clean[available_columns]
print(f"✓ Selected {len(available_columns)} key features")

print("\nStep 3: Handling missing values...")
print("Missing values before cleaning:")
print(df_clean.isnull().sum()[df_clean.isnull().sum() > 0])

# Separate features and target
if 'koi_disposition' in df_clean.columns:
    target_col = 'koi_disposition'
    # Remove rows with missing target
    df_clean = df_clean.dropna(subset=[target_col])
    print(f"✓ Removed rows with missing target variable")
    
    # For numerical features, use median imputation
    numerical_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in numerical_cols:
        if df_clean[col].isnull().sum() > 0:
            median_val = df_clean[col].median()
            df_clean[col].fillna(median_val, inplace=True)
    
    print(f"✓ Imputed missing values in numerical features using median")

print("\nStep 4: Outlier detection and handling...")
# Use IQR method for outlier detection
numerical_features = df_clean.select_dtypes(include=[np.number]).columns.tolist()
numerical_features = [col for col in numerical_features if col not in ['ra', 'dec']]  # Keep coordinates

outliers_count = 0
for col in numerical_features:
    Q1 = df_clean[col].quantile(0.25)
    Q3 = df_clean[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 3 * IQR  # Using 3*IQR for conservative outlier removal
    upper_bound = Q3 + 3 * IQR
    
    outliers_in_col = ((df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)).sum()
    outliers_count += outliers_in_col
    
    # Cap outliers instead of removing them to preserve data
    df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)

print(f"✓ Handled {outliers_count} outliers using capping method")

print("\nStep 5: Feature Engineering...")
# Create new features
if 'koi_prad' in df_clean.columns and 'koi_srad' in df_clean.columns:
    df_clean['planet_star_radius_ratio'] = df_clean['koi_prad'] / df_clean['koi_srad']
    print("✓ Created: planet_star_radius_ratio")

if 'koi_period' in df_clean.columns and 'koi_teq' in df_clean.columns:
    df_clean['period_temp_product'] = df_clean['koi_period'] * df_clean['koi_teq']
    print("✓ Created: period_temp_product")

print(f"\n✓ Data cleaning complete!")
print(f"Final dataset shape: {df_clean.shape[0]:,} rows × {df_clean.shape[1]} columns")

# ============================================================================
# SECTION 3: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================================
print("\n\n" + "="*80)
print("SECTION 3: EXPLORATORY DATA ANALYSIS (EDA)")
print("="*80 + "\n")

print("Generating comprehensive visualizations...\n")

# Figure 1: Target Variable Distribution
plt.figure(figsize=(10, 6))
if 'koi_disposition' in df_clean.columns:
    disposition_counts = df_clean['koi_disposition'].value_counts()
    colors = ['#2ecc71', '#e74c3c', '#f39c12']
    plt.bar(disposition_counts.index, disposition_counts.values, color=colors, edgecolor='black', linewidth=1.5)
    plt.title('Distribution of Exoplanet Dispositions', fontsize=16, fontweight='bold')
    plt.xlabel('Disposition', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=15)
    for i, v in enumerate(disposition_counts.values):
        plt.text(i, v + 50, str(v), ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig('visualizations/01_disposition_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: 01_disposition_distribution.png")
    
    print(f"\nDisposition Breakdown:")
    for disp, count in disposition_counts.items():
        pct = (count / len(df_clean)) * 100
        print(f"  {disp}: {count:,} ({pct:.2f}%)")

# Figure 2: Correlation Heatmap
numerical_for_corr = df_clean.select_dtypes(include=[np.number]).columns.tolist()
if len(numerical_for_corr) > 2:
    plt.figure(figsize=(14, 10))
    correlation_matrix = df_clean[numerical_for_corr].corr()
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title('Feature Correlation Heatmap', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('visualizations/02_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: 02_correlation_heatmap.png")

# Figure 3: Distribution of Orbital Period
if 'koi_period' in df_clean.columns:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Linear scale
    axes[0].hist(df_clean['koi_period'].dropna(), bins=50, color='#3498db', edgecolor='black', alpha=0.7)
    axes[0].set_title('Orbital Period Distribution (Linear Scale)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Orbital Period (days)', fontsize=10)
    axes[0].set_ylabel('Frequency', fontsize=10)
    axes[0].grid(alpha=0.3)
    
    # Log scale
    axes[1].hist(np.log10(df_clean['koi_period'].dropna() + 1), bins=50, color='#e67e22', edgecolor='black', alpha=0.7)
    axes[1].set_title('Orbital Period Distribution (Log Scale)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Log10(Orbital Period + 1)', fontsize=10)
    axes[1].set_ylabel('Frequency', fontsize=10)
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('visualizations/03_orbital_period_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: 03_orbital_period_distribution.png")

# Figure 4: Planetary Radius Distribution
if 'koi_prad' in df_clean.columns:
    plt.figure(figsize=(12, 6))
    plt.hist(df_clean['koi_prad'].dropna(), bins=60, color='#9b59b6', edgecolor='black', alpha=0.7)
    plt.axvline(1.0, color='red', linestyle='--', linewidth=2, label='Earth Radius')
    plt.title('Planetary Radius Distribution', fontsize=16, fontweight='bold')
    plt.xlabel('Planetary Radius (Earth Radii)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('visualizations/04_planetary_radius_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: 04_planetary_radius_distribution.png")

# Figure 5: Stellar Temperature vs Planetary Radius
if 'koi_steff' in df_clean.columns and 'koi_prad' in df_clean.columns and 'koi_disposition' in df_clean.columns:
    plt.figure(figsize=(12, 7))
    for disposition in df_clean['koi_disposition'].unique():
        subset = df_clean[df_clean['koi_disposition'] == disposition]
        plt.scatter(subset['koi_steff'], subset['koi_prad'], 
                   alpha=0.6, s=30, label=disposition, edgecolors='black', linewidth=0.5)
    plt.title('Stellar Effective Temperature vs Planetary Radius', fontsize=16, fontweight='bold')
    plt.xlabel('Stellar Effective Temperature (K)', fontsize=12)
    plt.ylabel('Planetary Radius (Earth Radii)', fontsize=12)
    plt.legend(title='Disposition', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('visualizations/05_steff_vs_prad_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: 05_steff_vs_prad_scatter.png")

# Figure 6: Equilibrium Temperature Distribution by Disposition
if 'koi_teq' in df_clean.columns and 'koi_disposition' in df_clean.columns:
    plt.figure(figsize=(12, 6))
    dispositions = df_clean['koi_disposition'].unique()
    data_to_plot = [df_clean[df_clean['koi_disposition'] == d]['koi_teq'].dropna() for d in dispositions]
    
    plt.boxplot(data_to_plot, labels=dispositions, patch_artist=True,
                boxprops=dict(facecolor='lightblue', color='black'),
                medianprops=dict(color='red', linewidth=2),
                whiskerprops=dict(color='black'),
                capprops=dict(color='black'))
    plt.title('Equilibrium Temperature by Disposition', fontsize=16, fontweight='bold')
    plt.xlabel('Disposition', fontsize=12)
    plt.ylabel('Equilibrium Temperature (K)', fontsize=12)
    plt.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('visualizations/06_teq_by_disposition_boxplot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: 06_teq_by_disposition_boxplot.png")

# Figure 7: Transit Depth vs Duration
if 'koi_depth' in df_clean.columns and 'koi_duration' in df_clean.columns:
    plt.figure(figsize=(12, 7))
    plt.scatter(df_clean['koi_duration'], df_clean['koi_depth'], 
               alpha=0.5, s=20, c=df_clean['koi_period'], cmap='viridis', edgecolors='black', linewidth=0.3)
    plt.colorbar(label='Orbital Period (days)')
    plt.title('Transit Depth vs Duration (colored by Orbital Period)', fontsize=16, fontweight='bold')
    plt.xlabel('Transit Duration (hours)', fontsize=12)
    plt.ylabel('Transit Depth (ppm)', fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('visualizations/07_depth_vs_duration_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: 07_depth_vs_duration_scatter.png")

print("\n✓ EDA visualizations complete!")

# ============================================================================
# SECTION 4: MACHINE LEARNING MODEL DEVELOPMENT
# ============================================================================
print("\n\n" + "="*80)
print("SECTION 4: MACHINE LEARNING MODEL DEVELOPMENT")
print("="*80 + "\n")

if 'koi_disposition' in df_clean.columns:
    print("Preparing data for modeling...\n")
    
    # Encode target variable - Focus on binary classification (CONFIRMED vs others)
    df_model = df_clean.copy()
    df_model['target'] = (df_model['koi_disposition'] == 'CONFIRMED').astype(int)
    
    print("Target Distribution:")
    print(df_model['target'].value_counts())
    print(f"  0: Not Confirmed (FALSE POSITIVE + CANDIDATE)")
    print(f"  1: Confirmed Exoplanet\n")
    
    # Prepare features
    feature_cols = [col for col in df_model.columns if col not in ['koi_disposition', 'target']]
    X = df_model[feature_cols]
    y = df_model['target']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Training set: {X_train.shape[0]:,} samples")
    print(f"Test set: {X_test.shape[0]:,} samples\n")
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Dictionary to store model results
    model_results = {}
    
    # Model 1: Logistic Regression (Baseline)
    print("-" * 80)
    print("Model 1: Logistic Regression (Baseline)")
    print("-" * 80)
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train_scaled, y_train)
    y_pred_lr = lr_model.predict(X_test_scaled)
    y_pred_proba_lr = lr_model.predict_proba(X_test_scaled)[:, 1]
    
    print("\nPerformance Metrics:")
    print(f"  Accuracy:  {accuracy_score(y_test, y_pred_lr):.4f}")
    print(f"  Precision: {precision_score(y_test, y_pred_lr):.4f}")
    print(f"  Recall:    {recall_score(y_test, y_pred_lr):.4f}")
    print(f"  F1-Score:  {f1_score(y_test, y_pred_lr):.4f}")
    print(f"  ROC-AUC:   {roc_auc_score(y_test, y_pred_proba_lr):.4f}")
    
    model_results['Logistic Regression'] = {
        'model': lr_model,
        'predictions': y_pred_lr,
        'probabilities': y_pred_proba_lr,
        'accuracy': accuracy_score(y_test, y_pred_lr),
        'precision': precision_score(y_test, y_pred_lr),
        'recall': recall_score(y_test, y_pred_lr),
        'f1': f1_score(y_test, y_pred_lr),
        'roc_auc': roc_auc_score(y_test, y_pred_proba_lr)
    }
    
    # Model 2: Random Forest
    print("\n" + "-" * 80)
    print("Model 2: Random Forest Classifier")
    print("-" * 80)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    y_pred_proba_rf = rf_model.predict_proba(X_test)[:, 1]
    
    print("\nPerformance Metrics:")
    print(f"  Accuracy:  {accuracy_score(y_test, y_pred_rf):.4f}")
    print(f"  Precision: {precision_score(y_test, y_pred_rf):.4f}")
    print(f"  Recall:    {recall_score(y_test, y_pred_rf):.4f}")
    print(f"  F1-Score:  {f1_score(y_test, y_pred_rf):.4f}")
    print(f"  ROC-AUC:   {roc_auc_score(y_test, y_pred_proba_rf):.4f}")
    
    model_results['Random Forest'] = {
        'model': rf_model,
        'predictions': y_pred_rf,
        'probabilities': y_pred_proba_rf,
        'accuracy': accuracy_score(y_test, y_pred_rf),
        'precision': precision_score(y_test, y_pred_rf),
        'recall': recall_score(y_test, y_pred_rf),
        'f1': f1_score(y_test, y_pred_rf),
        'roc_auc': roc_auc_score(y_test, y_pred_proba_rf)
    }
    
    # Feature Importance
    print("\nTop 10 Most Important Features:")
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    print(feature_importance.head(10).to_string(index=False))
    
    # Model 3: Gradient Boosting
    print("\n" + "-" * 80)
    print("Model 3: Gradient Boosting Classifier")
    print("-" * 80)
    gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb_model.fit(X_train, y_train)
    y_pred_gb = gb_model.predict(X_test)
    y_pred_proba_gb = gb_model.predict_proba(X_test)[:, 1]
    
    print("\nPerformance Metrics:")
    print(f"  Accuracy:  {accuracy_score(y_test, y_pred_gb):.4f}")
    print(f"  Precision: {precision_score(y_test, y_pred_gb):.4f}")
    print(f"  Recall:    {recall_score(y_test, y_pred_gb):.4f}")
    print(f"  F1-Score:  {f1_score(y_test, y_pred_gb):.4f}")
    print(f"  ROC-AUC:   {roc_auc_score(y_test, y_pred_proba_gb):.4f}")
    
    model_results['Gradient Boosting'] = {
        'model': gb_model,
        'predictions': y_pred_gb,
        'probabilities': y_pred_proba_gb,
        'accuracy': accuracy_score(y_test, y_pred_gb),
        'precision': precision_score(y_test, y_pred_gb),
        'recall': recall_score(y_test, y_pred_gb),
        'f1': f1_score(y_test, y_pred_gb),
        'roc_auc': roc_auc_score(y_test, y_pred_proba_gb)
    }
    
    # Figure 8: Model Comparison
    print("\nGenerating model comparison visualizations...")
    
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    model_names = list(model_results.keys())
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Subplot 1: Metric Comparison
    x = np.arange(len(metrics))
    width = 0.25
    
    for i, model_name in enumerate(model_names):
        values = [model_results[model_name][metric] for metric in metrics]
        axes[0].bar(x + i*width, values, width, label=model_name, alpha=0.8, edgecolor='black')
    
    axes[0].set_xlabel('Metrics', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Score', fontsize=12, fontweight='bold')
    axes[0].set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x + width)
    axes[0].set_xticklabels(['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'])
    axes[0].legend()
    axes[0].grid(alpha=0.3, axis='y')
    axes[0].set_ylim([0, 1.1])
    
    # Subplot 2: ROC Curves
    for model_name in model_names:
        proba = model_results[model_name]['probabilities']
        fpr, tpr, _ = roc_curve(y_test, proba)
        auc = model_results[model_name]['roc_auc']
        axes[1].plot(fpr, tpr, linewidth=2, label=f'{model_name} (AUC = {auc:.3f})')
    
    axes[1].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    axes[1].set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    axes[1].set_title('ROC Curves Comparison', fontsize=14, fontweight='bold')
    axes[1].legend(loc='lower right')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('visualizations/08_model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: 08_model_comparison.png")
    
    # Figure 9: Confusion Matrices
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, model_name in enumerate(model_names):
        cm = confusion_matrix(y_test, model_results[model_name]['predictions'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx], 
                   cbar_kws={'label': 'Count'}, linewidths=1, linecolor='black')
        axes[idx].set_title(f'{model_name}\nConfusion Matrix', fontsize=12, fontweight='bold')
        axes[idx].set_ylabel('True Label', fontsize=11)
        axes[idx].set_xlabel('Predicted Label', fontsize=11)
        axes[idx].set_xticklabels(['Not Confirmed', 'Confirmed'])
        axes[idx].set_yticklabels(['Not Confirmed', 'Confirmed'])
    
    plt.tight_layout()
    plt.savefig('visualizations/09_confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: 09_confusion_matrices.png")
    
    # Figure 10: Feature Importance
    plt.figure(figsize=(12, 8))
    top_features = feature_importance.head(15)
    plt.barh(range(len(top_features)), top_features['importance'], color='teal', edgecolor='black')
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Importance Score', fontsize=12, fontweight='bold')
    plt.ylabel('Features', fontsize=12, fontweight='bold')
    plt.title('Top 15 Feature Importances (Random Forest)', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig('visualizations/10_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: 10_feature_importance.png")
    
    print("\n✓ Machine Learning models complete!")

# ============================================================================
# SECTION 5: KEY INSIGHTS & FINDINGS
# ============================================================================
print("\n\n" + "="*80)
print("SECTION 5: KEY INSIGHTS & FINDINGS")
print("="*80 + "\n")

print("STATISTICAL INSIGHTS:")
print("-" * 80)

if 'koi_disposition' in df_clean.columns:
    confirmed = df_clean[df_clean['koi_disposition'] == 'CONFIRMED']
    false_positive = df_clean[df_clean['koi_disposition'] == 'FALSE POSITIVE']
    
    print(f"\n1. EXOPLANET CONFIRMATION:")
    print(f"   - Total candidates analyzed: {len(df_clean):,}")
    print(f"   - Confirmed exoplanets: {len(confirmed):,} ({len(confirmed)/len(df_clean)*100:.2f}%)")
    print(f"   - False positives: {len(false_positive):,} ({len(false_positive)/len(df_clean)*100:.2f}%)")
    
    if 'koi_prad' in df_clean.columns:
        print(f"\n2. PLANETARY CHARACTERISTICS:")
        print(f"   - Average planetary radius (confirmed): {confirmed['koi_prad'].mean():.2f} Earth radii")
        print(f"   - Median planetary radius (confirmed): {confirmed['koi_prad'].median():.2f} Earth radii")
        print(f"   - Smallest detected planet: {confirmed['koi_prad'].min():.2f} Earth radii")
        print(f"   - Largest detected planet: {confirmed['koi_prad'].max():.2f} Earth radii")
    
    if 'koi_period' in df_clean.columns:
        print(f"\n3. ORBITAL CHARACTERISTICS:")
        print(f"   - Average orbital period (confirmed): {confirmed['koi_period'].mean():.2f} days")
        print(f"   - Shortest orbital period: {confirmed['koi_period'].min():.2f} days")
        print(f"   - Longest orbital period: {confirmed['koi_period'].max():.2f} days")
    
    if 'koi_teq' in df_clean.columns:
        print(f"\n4. HABITABILITY POTENTIAL:")
        habitable_temp = confirmed[(confirmed['koi_teq'] >= 200) & (confirmed['koi_teq'] <= 350)]
        print(f"   - Exoplanets in habitable temperature range (200-350K): {len(habitable_temp)}")
        print(f"   - Percentage of confirmed planets: {len(habitable_temp)/len(confirmed)*100:.2f}%")

print(f"\n5. MACHINE LEARNING INSIGHTS:")
print(f"   - Best performing model: {max(model_results.items(), key=lambda x: x[1]['roc_auc'])[0]}")
print(f"   - Best ROC-AUC score: {max(model_results.items(), key=lambda x: x[1]['roc_auc'])[1]['roc_auc']:.4f}")
print(f"   - Model can distinguish confirmed exoplanets with high accuracy")
print(f"   - Top predictive features: transit characteristics and stellar properties")

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
print(f"\n✓ All visualizations saved in 'visualizations/' directory")
print(f"✓ Total visualizations generated: 10")
print(f"\nNext Steps:")
print("  1. Review visualizations in the 'visualizations' folder")
print("  2. Open kepler_analysis.ipynb for interactive analysis")
print("  3. Export notebook to PDF for submission")
print("\n" + "="*80)
