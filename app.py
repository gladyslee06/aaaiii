# ==========================
# Import Libraries
# ==========================
import pandas as pd
import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neighbors import LocalOutlierFactor

# Models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# Evaluation
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc
)

# ==========================
# Setup Plot Folder
# ==========================
PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# Set style for better looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10

# ==========================
# Helper Functions
# ==========================
def display_dataframe(df, title="DataFrame", n_rows=10):
    """Display dataframe with proper formatting"""
    print(f"\n{'='*60}")
    print(f"{title}")
    print('='*60)
    print(df.head(n_rows).to_string())
    print(f"\nShape: {df.shape}")

def display_statistics(df, title="Summary Statistics"):
    """Display comprehensive statistics for a dataframe"""
    print(f"\n{'-'*100}")
    print(f"SUMMARY STATISTICS: {title}")
    print('-'*100)

    stats = df.describe().T
    stats['median'] = df.median()
    stats['missing'] = df.isnull().sum()
    stats['missing_pct'] = (df.isnull().sum() / len(df) * 100).round(2)

    stats = stats[[
        'count', 'mean', 'std', 'min', '25%', 'median', '50%', '75%', 'max', 'missing', 'missing_pct'
    ]]

    stats.columns = [
        'Count', 'Mean', 'Std Dev', 'Min', 'Q1 (25%)', 'Median',
        'Q2 (50%)', 'Q3 (75%)', 'Max', 'Missing', 'Missing %'
    ]

    numeric_cols = ['Mean', 'Std Dev', 'Min', 'Q1 (25%)', 'Median', 'Q2 (50%)', 'Q3 (75%)', 'Max']
    stats[numeric_cols] = stats[numeric_cols].round(2)

    print(stats.to_string())
    return stats

def evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    """Evaluate model on both train and test sets to check overfitting"""
    # Training set evaluation
    y_train_pred = model.predict(X_train)
    train_metrics = {
        'accuracy': accuracy_score(y_train, y_train_pred),
        'precision': precision_score(y_train, y_train_pred,zero_division=0),
        'recall': recall_score(y_train, y_train_pred,zero_division=0),
        'f1': f1_score(y_train, y_train_pred),
        'y_pred': y_train_pred
    }

    # Test set evaluation
    y_test_pred = model.predict(X_test)
    test_metrics = {
        'accuracy': accuracy_score(y_test, y_test_pred),
        'precision': precision_score(y_test, y_test_pred,zero_division=0),
        'recall': recall_score(y_test, y_test_pred,zero_division=0),
        'f1': f1_score(y_test, y_test_pred),
        'y_pred': y_test_pred
    }

    # Calculate overfitting gap
    overfitting_gap = train_metrics['accuracy'] - test_metrics['accuracy']

    return train_metrics, test_metrics, overfitting_gap


# ==========================
# COMMON DATA PREPARATION (Shared by all models)
# ==========================
print("\n" + "="*80)
print("DIABETES CLASSIFICATION - COMMON DATA PREPARATION")
print("="*80)

try:
    df = pd.read_csv("diabetes.csv")
except FileNotFoundError:
    print("Error: diabetes.csv not found!")
    print("Please ensure the file is in the current directory.")
    exit(1)

print("\nDataset Information:")
print(df.info())

# ==========================
# EDA - DATA EXPLORATION
# ==========================
print("\n" + "="*60)
print("EXPLORATORY DATA ANALYSIS")
print("="*60)

display_dataframe(df, "DIABETES DATASET (First 10 Rows)")

print(f"\nDataset Shape: {df.shape}")
print(f"Number of Features: {len(df.columns)-1}")
print(f"Number of Records: {len(df)}")

print("\nTarget Distribution:")
print(f"  Non-Diabetic (0): {sum(df['Outcome']==0)} ({sum(df['Outcome']==0)/len(df)*100:.1f}%)")
print(f"  Diabetic (1):     {sum(df['Outcome']==1)} ({sum(df['Outcome']==1)/len(df)*100:.1f}%)")

display_statistics(df, "Original Dataset (Before Preprocessing)")

# ==========================
# GRAPH 1: Target Class Distribution
# ==========================
fig, ax = plt.subplots(figsize=(7, 5))
colors = ['#4C72B0', '#C44E52']
sns.countplot(x="Outcome", data=df, hue="Outcome", palette=colors, legend=False)

ax.set_xticks([0, 1])
ax.set_xticklabels(["Non-Diabetic (0)", "Diabetic (1)"])
ax.set_title("Distribution of Diabetes Cases in the Dataset", fontsize=14, fontweight='bold')
ax.set_xlabel("Patient Classification", fontsize=12)
ax.set_ylabel("Number of Patients", fontsize=12)

for container in ax.containers:
    ax.bar_label(container, fontsize=11, fontweight='bold')

ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "01_class_distribution.png"), dpi=150, bbox_inches='tight')
plt.show()
plt.close()
print("Created: 01_class_distribution.png")

# ==========================
# HANDLE MISSING VALUES (Zero Values)
# ==========================
cols_with_missing = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

zero_counts = df[cols_with_missing].apply(lambda col: (col == 0).sum())

print("\n" + "="*60)
print("MISSING VALUES (ZERO VALUES) ANALYSIS")
print("="*60)
print(zero_counts)

# ==========================
# GRAPH 2: Missing Values Visualization
# ==========================
fig, ax = plt.subplots(figsize=(9, 6))
bars = sns.barplot(
    x=zero_counts.index,
    y=zero_counts.values,
    hue=zero_counts.index,
    palette="Blues_d",
    legend=False
)

ax.set_title("Missing Values Per Clinical Feature Before Imputation",
             fontsize=14, fontweight='bold')
ax.set_xlabel("Clinical Features", fontsize=12)
ax.set_ylabel("Number of Missing (Zero) Values", fontsize=12)

for container in ax.containers:
    ax.bar_label(container, fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "02_missing_values.png"), dpi=150, bbox_inches='tight')
plt.show()
plt.close()
print("Created: 02_missing_values.png")

# ==========================
# SPLIT DATA FIRST (PREVENT DATA LEAKAGE)
# ==========================
print("\n" + "="*60)
print("DATA SPLIT (BEFORE ANY PREPROCESSING)")
print("="*60)

X = df.drop("Outcome", axis=1).copy()
y = df["Outcome"].copy()

X[cols_with_missing] = X[cols_with_missing].replace(0, np.nan)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {len(X_train)} samples")
print(f"  - Non-Diabetic: {sum(y_train==0)} ({sum(y_train==0)/len(y_train)*100:.1f}%)")
print(f"  - Diabetic: {sum(y_train==1)} ({sum(y_train==1)/len(y_train)*100:.1f}%)")
print(f"Test set: {len(X_test)} samples")
print(f"  - Non-Diabetic: {sum(y_test==0)} ({sum(y_test==0)/len(y_test)*100:.1f}%)")
print(f"  - Diabetic: {sum(y_test==1)} ({sum(y_test==1)/len(y_test)*100:.1f}%)")

# ==========================
# IMPUTATION (FIT ON TRAIN ONLY)
# ==========================
print("\n" + "="*60)
print("IMPUTATION (FIT ON TRAINING DATA ONLY)")
print("="*60)

imputer = SimpleImputer(strategy="median")
X_train_imputed = X_train.copy()
X_test_imputed = X_test.copy()

X_train_imputed[cols_with_missing] = imputer.fit_transform(X_train[cols_with_missing])
X_test_imputed[cols_with_missing] = imputer.transform(X_test[cols_with_missing])

print("Imputation complete - using median values from training data")
print(f"Missing values in training: {X_train_imputed.isnull().sum().sum()}")
print(f"Missing values in test: {X_test_imputed.isnull().sum().sum()}")

print("\n" + "="*60)
print("STATISTICS AFTER IMPUTATION")
print("="*60)
display_statistics(X_train_imputed, "Training Data (After Imputation)")
display_statistics(X_test_imputed, "Test Data (After Imputation)")

# ==========================
# CORRELATION ANALYSIS (Using Training Data)
# ==========================
print("\n" + "="*60)
print("FEATURE CORRELATION WITH DIABETES OUTCOME")
print("="*60)

train_with_target = X_train_imputed.copy()
train_with_target['Outcome'] = y_train.values
correlation_with_target = train_with_target.corr()['Outcome'].sort_values(ascending=False)

print(correlation_with_target.round(4))

print("\nTop 3 features most correlated with diabetes:")
for i in range(1, min(4, len(correlation_with_target))):
    feature = correlation_with_target.index[i]
    value = correlation_with_target.iloc[i]
    print(f"  {i}. {feature}: {value:.4f}")

# ==========================
# GRAPH 3: Correlation Heatmap
# ==========================
fig, ax = plt.subplots(figsize=(10, 8))
corr = train_with_target.corr()

heatmap = sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    square=True,
    linewidths=1.0,
    linecolor='white',
    cbar_kws={"shrink": 0.8, "label": "Correlation Coefficient"},
    annot_kws={"size": 10}
)

ax.set_title("Feature Correlation Matrix (Training Data)", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "03_correlation_heatmap.png"), dpi=150, bbox_inches='tight')
plt.show()
plt.close()
print("Created: 03_correlation_heatmap.png")

# ==========================
# GRAPH 4: Feature Distributions by Outcome
# ==========================
feature_cols = [c for c in X_train_imputed.columns if c != "Outcome"]

fig, axes = plt.subplots(2, 4, figsize=(16, 9))
axes = axes.flatten()

for i, col in enumerate(feature_cols):
    plot_data = X_train_imputed.copy()
    plot_data['Outcome'] = y_train.values

    sns.histplot(
        data=plot_data,
        x=col,
        hue="Outcome",
        kde=True,
        palette=["#4C72B0", "#C44E52"],
        alpha=0.6,
        ax=axes[i],
        stat='density',
        common_norm=False
    )

    axes[i].set_title(col, fontsize=11, fontweight='bold')
    axes[i].set_xlabel('')
    axes[i].grid(True, alpha=0.3)

    if i == 0:
        axes[i].legend(title='Outcome', labels=['Non-Diabetic', 'Diabetic'])

fig.suptitle("Distribution of Clinical Features by Diabetes Status",
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "04_feature_distributions.png"), dpi=150, bbox_inches='tight')
plt.show()
plt.close()
print("Created: 04_feature_distributions.png")

# ==========================
# GRAPH 5: Box Plot Before Standardization
# ==========================
data_melted_before = pd.melt(
    X_train_imputed.assign(Outcome=y_train.values),
    id_vars='Outcome',
    var_name='Features',
    value_name='Value'
)

fig, ax = plt.subplots(figsize=(14, 7))
sns.boxplot(
    x='Features',
    y='Value',
    hue='Outcome',
    data=data_melted_before,
    palette=['#4C72B0', '#C44E52']
)

plt.title('Feature Distribution by Outcome (Training Data - Before Outlier Treatment)',
          fontsize=14, fontweight='bold')
plt.xticks(rotation=0)
plt.xlabel('Clinical Features', fontsize=12)
plt.ylabel('Value', fontsize=12)
plt.legend(title='Outcome', labels=['Non-Diabetic', 'Diabetic'])
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "05_boxplot_before_outlier_treatment.png"), dpi=150, bbox_inches='tight')
plt.show()
plt.close()
print("Created: 05_boxplot_before_outlier_treatment.png")

# ==========================
# OUTLIER DETECTION & TREATMENT (IMPUTE WITH MEDIAN)
# ==========================
print("\n" + "="*60)
print("OUTLIER DETECTION & TREATMENT (IMPUTE WITH MEDIAN)")
print("="*60)

lof = LocalOutlierFactor(contamination='auto', n_neighbors=20)
y_pred_train = lof.fit_predict(X_train_imputed)
outlier_mask = y_pred_train == -1
outlier_indices = np.where(outlier_mask)[0]

print(f"Number of outliers detected in training data: {len(outlier_indices)}")
print(f"Outlier percentage: {len(outlier_indices)/len(X_train_imputed)*100:.2f}%")

X_train_clean = X_train_imputed.copy()
y_train_clean = y_train.copy()

if len(outlier_indices) > 0:
    median_values = X_train_imputed[~outlier_mask].median()

    print("\nImputing outliers with median values...")
    print("Median values used for imputation:")
    for col in feature_cols:
        print(f"  {col}: {median_values[col]:.2f}")

    for col in feature_cols:
        X_train_clean.loc[outlier_mask, col] = median_values[col]

    print(f"\nOutliers treated: {len(outlier_indices)} samples imputed with median values")
    print(f"Training data shape maintained: {X_train_clean.shape}")
else:
    print("No outliers detected. Data unchanged.")

print(f"\nTraining data after outlier treatment:")
print(f"  - Samples: {len(X_train_clean)} (unchanged)")
print(f"  - Non-Diabetic: {sum(y_train_clean==0)} ({sum(y_train_clean==0)/len(y_train_clean)*100:.1f}%)")
print(f"  - Diabetic: {sum(y_train_clean==1)} ({sum(y_train_clean==1)/len(y_train_clean)*100:.1f}%)")

display_statistics(X_train_clean, "Training Data (After Outlier Treatment - Imputed with Median)")

# ==========================
# GRAPH 6: Outlier Visualization (Before vs After)
# ==========================
if len(outlier_indices) > 0:
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    axes[0].scatter(
        X_train_imputed.iloc[~outlier_mask, 0],
        X_train_imputed.iloc[~outlier_mask, 1],
        color='blue', s=20, label='Normal Points', alpha=0.4
    )
    axes[0].scatter(
        X_train_imputed.iloc[outlier_indices, 0],
        X_train_imputed.iloc[outlier_indices, 1],
        color='red', s=80, label='Outliers', edgecolors='white', linewidth=1.5, alpha=0.7
    )
    axes[0].set_xlabel(feature_cols[0], fontsize=12)
    axes[0].set_ylabel(feature_cols[1], fontsize=12)
    axes[0].set_title('Before Outlier Treatment', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(
        X_train_clean.iloc[~outlier_mask, 0],
        X_train_clean.iloc[~outlier_mask, 1],
        color='blue', s=20, label='Normal Points', alpha=0.4
    )
    axes[1].scatter(
        X_train_clean.iloc[outlier_indices, 0],
        X_train_clean.iloc[outlier_indices, 1],
        color='green', s=80, label='Imputed Outliers', edgecolors='white', linewidth=1.5, alpha=0.7
    )
    axes[1].set_xlabel(feature_cols[0], fontsize=12)
    axes[1].set_ylabel(feature_cols[1], fontsize=12)
    axes[1].set_title('After Outlier Treatment (Imputed with Median)', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('Outlier Detection and Treatment using Local Outlier Factor',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "06_outlier_treatment.png"), dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    print("Created: 06_outlier_treatment.png")

# ==========================
# SCALE DATA (FIT ON TRAINING DATA ONLY)
# ==========================
print("\n" + "="*60)
print("SCALING DATA (FIT ON TRAINING DATA ONLY)")
print("="*60)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_clean)
X_test_scaled = scaler.transform(X_test_imputed)

print(" Scaler fitted on training data and applied to both train and test sets")

# ==========================
# GRAPH 7: Box Plot After Scaling
# ==========================
X_scaled_df = pd.DataFrame(X_train_scaled, columns=feature_cols)
X_scaled_df['Outcome'] = y_train_clean.values

data_melted_after = pd.melt(
    X_scaled_df,
    id_vars='Outcome',
    var_name='Features',
    value_name='Value'
)

fig, ax = plt.subplots(figsize=(14, 7))
sns.boxplot(
    x='Features',
    y='Value',
    hue='Outcome',
    data=data_melted_after,
    palette=['#4C72B0', '#C44E52']
)

plt.title('Feature Distribution by Outcome (Training Data - After Scaling)',
          fontsize=14, fontweight='bold')
plt.xticks(rotation=0)
plt.xlabel('Clinical Features', fontsize=12)
plt.ylabel('Scaled Value', fontsize=12)
plt.legend(title='Outcome', labels=['Non-Diabetic', 'Diabetic'])
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "07_boxplot_after_scaling.png"), dpi=150, bbox_inches='tight')
plt.show()
plt.close()
print("Created: 07_boxplot_after_scaling.png")

print("\n" + "="*80)
print("COMMON DATA PREPARATION COMPLETE")
print(f"Training data ready: {len(X_train_scaled)} samples (scaled)")
print(f"Test data ready: {len(X_test_scaled)} samples (scaled)")
print("="*80)


# ##################################################################
# MODEL 1: K-NEAREST NEIGHBORS (KNN) - Lau Kai Hang
# ##################################################################
print("\n" + "="*80)
print("MODEL 1: K-NEAREST NEIGHBORS (KNN) - REGULARIZED")
print("="*80)

print("\n1. Creating Regularized KNN Model")
knn_model = KNeighborsClassifier(
    n_neighbors=15,               # INCREASED from 7 to prevent overfitting
    weights='distance',
    metric='minkowski',
    p=2,
    leaf_size=30,
    n_jobs=-1
)
print("   Model: KNeighborsClassifier")
print(f"   - n_neighbors: 15 (increased from 7 for simpler model)")
print(f"   - weights: 'distance'")

print("\n2. Training KNN on Full Training Data")
knn_model.fit(X_train_scaled, y_train_clean)
print("   Training complete!")

print("\n3. Evaluating KNN Model (Training vs Test)")
train_metrics_knn, test_metrics_knn, overfitting_gap_knn = evaluate_model(
    knn_model, X_train_scaled, y_train_clean, X_test_scaled, y_test, "KNN"
)

print(f"\n    KNN - Training Performance:")
print(f"      Training Accuracy:  {train_metrics_knn['accuracy']:.4f}")
print(f"      Training Precision: {train_metrics_knn['precision']:.4f}")
print(f"      Training Recall:    {train_metrics_knn['recall']:.4f}")
print(f"      Training F1 Score:  {train_metrics_knn['f1']:.4f}")

print(f"\n    KNN - Test Performance:")
print(f"      Test Accuracy:  {test_metrics_knn['accuracy']:.4f}")
print(f"      Test Precision: {test_metrics_knn['precision']:.4f}")
print(f"      Test Recall:    {test_metrics_knn['recall']:.4f}")
print(f"      Test F1 Score:  {test_metrics_knn['f1']:.4f}")

print(f"\n     Overfitting Gap (Train - Test): {overfitting_gap_knn:.4f}")
if overfitting_gap_knn > 0.05:
    print(f"        WARNING: Possible overfitting (gap > 0.05)")
else:
    print(f"       Good generalization (gap ≤ 0.05)")

print("\n4. Classification Report (Test Set)")
print(classification_report(y_test, test_metrics_knn['y_pred'], target_names=["Non-Diabetic", "Diabetic"]))

print("\n5. Confusion Matrix")
cm_knn = confusion_matrix(y_test, test_metrics_knn['y_pred'])
print(f"   [[{cm_knn[0,0]:4d} {cm_knn[0,1]:4d}]")
print(f"    [{cm_knn[1,0]:4d} {cm_knn[1,1]:4d}]]")

tn, fp, fn, tp = cm_knn.ravel()
specificity_knn = tn / (tn + fp)
npv_knn = tn / (tn + fn)

print(f"\n   Additional Metrics:")
print(f"   Specificity: {specificity_knn:.4f}")
print(f"   Negative Predictive Value: {npv_knn:.4f}")

print("\n6. ROC Curve Analysis")
y_proba_knn = knn_model.predict_proba(X_test_scaled)[:, 1]
fpr_knn, tpr_knn, _ = roc_curve(y_test, y_proba_knn)
roc_auc_knn = auc(fpr_knn, tpr_knn)
print(f"   AUC: {roc_auc_knn:.4f}")

print("\n7. Creating KNN Visualizations")

fig, ax = plt.subplots(figsize=(7, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm_knn, display_labels=["Non-Diabetic", "Diabetic"])
disp.plot(cmap="Blues", ax=ax, colorbar=False)
ax.set_title("KNN - Confusion Matrix (Test Set)", fontsize=14, fontweight='bold')
ax.set_xlabel("Predicted Label", fontsize=12)
ax.set_ylabel("True Label", fontsize=12)
ax.grid(False)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "knn_01_confusion_matrix.png"), dpi=150, bbox_inches='tight')
plt.show()
plt.close()
print("   Created: knn_01_confusion_matrix.png")

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(fpr_knn, tpr_knn, label=f"KNN (AUC = {roc_auc_knn:.3f})", linewidth=2.5, color='#2E86AB')
ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Guess (AUC = 0.5)", linewidth=1.5, alpha=0.7)
ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=12)
ax.set_title("KNN - ROC Curve", fontsize=14, fontweight='bold')
ax.legend(loc="lower right", fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_facecolor('#f8f9fa')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "knn_02_roc_curve.png"), dpi=150, bbox_inches='tight')
plt.show()
plt.close()
print("   Created: knn_02_roc_curve.png")

print("\n8. Saving KNN Model")
pickle.dump(knn_model, open("knn_model.pkl", "wb"))
print("   Model saved as 'knn_model.pkl'")

print("\n" + "="*80)
print("KNN MODEL COMPLETE")
print(f"Test Accuracy: {test_metrics_knn['accuracy']:.4f}")
print(f"Train Accuracy: {train_metrics_knn['accuracy']:.4f}")
print(f"Overfitting Gap: {overfitting_gap_knn:.4f}")
print("="*80)


# ##################################################################
# MODEL 2: SUPPORT VECTOR MACHINE (SVM) - Ng Kai Seng
# ##################################################################
print("\n" + "="*80)
print("MODEL 2: SUPPORT VECTOR MACHINE (SVM)")
print("="*80)

print("\n1. Creating Regularized SVM Model")
svm_model = SVC(
    kernel="rbf",
    C=0.1,                       # Lower C = more regularization
    gamma='scale',               # Auto-adjusted gamma
    class_weight='balanced',     # Handle class imbalance
    probability=True,
    random_state=42,
    shrinking=True
)
print("   Model: SVC with regularization")
print(f"   - C: 0.1 (lower = more regularization)")
print(f"   - gamma: 'scale'")
print(f"   - class_weight: 'balanced'")

print("\n2. Training SVM on Full Training Data")
svm_model.fit(X_train_scaled, y_train_clean)
print("   Training complete!")

print("\n3. Evaluating SVM Model (Training vs Test)")
train_metrics_svm, test_metrics_svm, overfitting_gap_svm = evaluate_model(
    svm_model, X_train_scaled, y_train_clean, X_test_scaled, y_test, "SVM"
)

print(f"\n    SVM - Training Performance:")
print(f"      Training Accuracy:  {train_metrics_svm['accuracy']:.4f}")
print(f"      Training Precision: {train_metrics_svm['precision']:.4f}")
print(f"      Training Recall:    {train_metrics_svm['recall']:.4f}")
print(f"      Training F1 Score:  {train_metrics_svm['f1']:.4f}")

print(f"\n    SVM - Test Performance:")
print(f"      Test Accuracy:  {test_metrics_svm['accuracy']:.4f}")
print(f"      Test Precision: {test_metrics_svm['precision']:.4f}")
print(f"      Test Recall:    {test_metrics_svm['recall']:.4f}")
print(f"      Test F1 Score:  {test_metrics_svm['f1']:.4f}")

print(f"\n     Overfitting Gap (Train - Test): {overfitting_gap_svm:.4f}")
if overfitting_gap_svm > 0.05:
    print(f"        WARNING: Possible overfitting (gap > 0.05)")
else:
    print(f"       Good generalization (gap ≤ 0.05)")

print("\n4. Classification Report (Test Set)")
print(classification_report(y_test, test_metrics_svm['y_pred'], target_names=["Non-Diabetic", "Diabetic"]))

print("\n5. Confusion Matrix")
cm_svm = confusion_matrix(y_test, test_metrics_svm['y_pred'])
print(f"   [[{cm_svm[0,0]:4d} {cm_svm[0,1]:4d}]")
print(f"    [{cm_svm[1,0]:4d} {cm_svm[1,1]:4d}]]")

tn, fp, fn, tp = cm_svm.ravel()
specificity_svm = tn / (tn + fp)
npv_svm = tn / (tn + fn)

print(f"\n   Additional Metrics:")
print(f"   Specificity: {specificity_svm:.4f}")
print(f"   Negative Predictive Value: {npv_svm:.4f}")

print("\n6. ROC Curve Analysis")
y_proba_svm = svm_model.predict_proba(X_test_scaled)[:, 1]
fpr_svm, tpr_svm, _ = roc_curve(y_test, y_proba_svm)
roc_auc_svm = auc(fpr_svm, tpr_svm)
print(f"   AUC: {roc_auc_svm:.4f}")

print("\n7. Creating SVM Visualizations")

fig, ax = plt.subplots(figsize=(7, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm_svm, display_labels=["Non-Diabetic", "Diabetic"])
disp.plot(cmap="Blues", ax=ax, colorbar=False)
ax.set_title("SVM - Confusion Matrix (Test Set)", fontsize=14, fontweight='bold')
ax.set_xlabel("Predicted Label", fontsize=12)
ax.set_ylabel("True Label", fontsize=12)
ax.grid(False)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "svm_01_confusion_matrix.png"), dpi=150, bbox_inches='tight')
plt.show()
plt.close()
print("   Created: svm_01_confusion_matrix.png")

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(fpr_svm, tpr_svm, label=f"SVM (AUC = {roc_auc_svm:.3f})", linewidth=2.5, color='#A23B72')
ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Guess (AUC = 0.5)", linewidth=1.5, alpha=0.7)
ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=12)
ax.set_title("SVM - ROC Curve", fontsize=14, fontweight='bold')
ax.legend(loc="lower right", fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_facecolor('#f8f9fa')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "svm_02_roc_curve.png"), dpi=150, bbox_inches='tight')
plt.show()
plt.close()
print("   Created: svm_02_roc_curve.png")

print("\n8. Saving SVM Model")
pickle.dump(svm_model, open("svm_model.pkl", "wb"))
print("   Model saved as 'svm_model.pkl'")

print("\n" + "="*80)
print("SVM MODEL COMPLETE")
print(f"Test Accuracy: {test_metrics_svm['accuracy']:.4f}")
print(f"Train Accuracy: {train_metrics_svm['accuracy']:.4f}")
print(f"Overfitting Gap: {overfitting_gap_svm:.4f}")
print("="*80)


# ##################################################################
# MODEL 3: RANDOM FOREST (RF) -  Gladys lee
# ##################################################################
print("\n" + "="*80)
print("MODEL 3: RANDOM FOREST (RF)")
print("="*80)

print("\n1. Creating Random Forest Model")
rf_model = RandomForestClassifier(
    random_state=42,
    n_estimators=50,
    max_depth=5,
    min_samples_split=20,
    min_samples_leaf=10,
    max_features='sqrt',
    min_impurity_decrease=0.01,
    ccp_alpha=0.005,
    class_weight='balanced',
    n_jobs=-1
)
print("   Model: RandomForestClassifier with regularization parameters")
print(f"   - n_estimators: 50 (reduced from 100)")
print(f"   - max_depth: 5 (prevents overfitting)")
print(f"   - min_samples_split: 20 (higher = less overfitting)")
print(f"   - min_samples_leaf: 10 (higher = less overfitting)")
print(f"   - ccp_alpha: 0.005 (pruning)")

print("\n2. Training Random Forest on Unscaled Training Data")
rf_model.fit(X_train_clean, y_train_clean)
print("   Training complete!")

print("\n3. Evaluating Random Forest Model (Training vs Test)")
train_metrics_rf, test_metrics_rf, overfitting_gap_rf = evaluate_model(
    rf_model,
    X_train_clean,
    y_train_clean,
    X_test_imputed,
    y_test,
    "Random Forest"
)

print(f"\n    Random Forest - Training Performance:")
print(f"      Training Accuracy:  {train_metrics_rf['accuracy']:.4f}")
print(f"      Training Precision: {train_metrics_rf['precision']:.4f}")
print(f"      Training Recall:    {train_metrics_rf['recall']:.4f}")
print(f"      Training F1 Score:  {train_metrics_rf['f1']:.4f}")

print(f"\n    Random Forest - Test Performance:")
print(f"      Test Accuracy:  {test_metrics_rf['accuracy']:.4f}")
print(f"      Test Precision: {test_metrics_rf['precision']:.4f}")
print(f"      Test Recall:    {test_metrics_rf['recall']:.4f}")
print(f"      Test F1 Score:  {test_metrics_rf['f1']:.4f}")

print(f"\n     Overfitting Gap (Train - Test): {overfitting_gap_rf:.4f}")
if overfitting_gap_rf > 0.05:
    print(f"        WARNING: Possible overfitting (gap > 0.05)")
else:
    print(f"       Good generalization (gap ≤ 0.05)")

print("\n4. Classification Report (Test Set)")
print(classification_report(y_test, test_metrics_rf['y_pred'], target_names=["Non-Diabetic", "Diabetic"]))

print("\n5. Confusion Matrix")
cm_rf = confusion_matrix(y_test, test_metrics_rf['y_pred'])
print(f"   [[{cm_rf[0,0]:4d} {cm_rf[0,1]:4d}]")
print(f"    [{cm_rf[1,0]:4d} {cm_rf[1,1]:4d}]]")

tn, fp, fn, tp = cm_rf.ravel()
specificity_rf = tn / (tn + fp)
npv_rf = tn / (tn + fn)

print(f"\n   Additional Metrics:")
print(f"   Specificity: {specificity_rf:.4f}")
print(f"   Negative Predictive Value: {npv_rf:.4f}")

print("\n6. ROC Curve Analysis")
y_proba_rf = rf_model.predict_proba(X_test_imputed)[:, 1]
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_proba_rf)
roc_auc_rf = auc(fpr_rf, tpr_rf)
print(f"   AUC: {roc_auc_rf:.4f}")

print("\n7. Feature Importance Analysis")
importances = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)
for rank, (feature, importance) in enumerate(importances.items(), 1):
    print(f"   {rank}. {feature}: {importance:.4f}")

print("\n8. Creating Random Forest Visualizations")

fig, ax = plt.subplots(figsize=(7, 6))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm_rf,
    display_labels=["Non-Diabetic", "Diabetic"]
)

disp.plot(
    cmap="Blues",
    ax=ax,
    colorbar=False
)

ax.set_title(
    "Random Forest - Confusion Matrix (Test Set)",
    fontsize=14,
    fontweight='bold'
)

ax.set_xlabel("Predicted Label", fontsize=12)
ax.set_ylabel("True Label", fontsize=12)

ax.grid(False)

plt.tight_layout()

plt.savefig(
    os.path.join(PLOT_DIR, "rf_01_confusion_matrix.png"),
    dpi=150,
    bbox_inches='tight'
)

plt.show()
plt.close()

print("   Created: rf_01_confusion_matrix.png")

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC = {roc_auc_rf:.3f})", linewidth=2.5, color='#F18F01')
ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Guess (AUC = 0.5)", linewidth=1.5, alpha=0.7)
ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=12)
ax.set_title("Random Forest - ROC Curve", fontsize=14, fontweight='bold')
ax.legend(loc="lower right", fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_facecolor('#f8f9fa')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "rf_02_roc_curve.png"), dpi=150, bbox_inches='tight')
plt.show()
plt.close()
print("   Created: rf_02_roc_curve.png")

importances_sorted = importances.sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(9, 6))
colors_import = plt.cm.RdYlGn_r(np.linspace(0, 0.8, len(importances_sorted)))
importances_sorted.plot(kind="barh", ax=ax, color=colors_import)
ax.set_title("Random Forest - Feature Importance Analysis", fontsize=14, fontweight='bold')
ax.set_xlabel("Importance Score", fontsize=12)
ax.set_ylabel("Clinical Features", fontsize=12)
for i, (idx, value) in enumerate(importances_sorted.items()):
    ax.text(value + 0.005, i, f"{value:.3f}", va='center', fontsize=10, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "rf_03_feature_importance.png"), dpi=150, bbox_inches='tight')
plt.show()
plt.close()
print("   Created: rf_03_feature_importance.png")

print("\n9. Saving Random Forest Model")
pickle.dump(rf_model, open("rf_model.pkl", "wb"))
print("   Model saved as 'rf_model.pkl'")

print("\n" + "="*80)
print("RANDOM FOREST MODEL COMPLETE")
print(f"Test Accuracy: {test_metrics_rf['accuracy']:.4f}")
print(f"Train Accuracy: {train_metrics_rf['accuracy']:.4f}")
print(f"Overfitting Gap: {overfitting_gap_rf:.4f}")
print("="*80)


# ##################################################################
# MODEL COMPARISON & FINAL EVALUATION
# ##################################################################
print("\n" + "="*80)
print("MODEL COMPARISON SUMMARY")
print("="*80)

comparison_df = pd.DataFrame({
    'KNN': [
        train_metrics_knn['accuracy'], test_metrics_knn['accuracy'],
        overfitting_gap_knn,
        test_metrics_knn['precision'], test_metrics_knn['recall'],
        test_metrics_knn['f1'],
        specificity_knn, npv_knn,
        roc_auc_knn
    ],
    'SVM': [
        train_metrics_svm['accuracy'], test_metrics_svm['accuracy'],
        overfitting_gap_svm,
        test_metrics_svm['precision'], test_metrics_svm['recall'],
        test_metrics_svm['f1'],
        specificity_svm, npv_svm,
        roc_auc_svm
    ],
    'Random Forest': [
        train_metrics_rf['accuracy'], test_metrics_rf['accuracy'],
        overfitting_gap_rf,
        test_metrics_rf['precision'], test_metrics_rf['recall'],
        test_metrics_rf['f1'],
        specificity_rf, npv_rf,
        roc_auc_rf
    ]
}, index=[
    "Train Accuracy", "Test Accuracy", "Overfitting Gap",
    "Test Precision", "Test Recall", "Test F1 Score",
    "Specificity", "Negative Predictive Value",
    "ROC AUC"
])

print("\nComprehensive Performance Comparison:")
print(comparison_df.round(4))

best_model_name = comparison_df.loc['Test Accuracy'].idxmax()
best_accuracy = comparison_df.loc['Test Accuracy'].max()

print(f"\n{'='*80}")
print(f" BEST MODEL (by Test Accuracy): {best_model_name}")
print(f"  Test Accuracy: {best_accuracy:.4f}")
print(f"  Train Accuracy: {comparison_df.loc['Train Accuracy', best_model_name]:.4f}")
print(f"  Overfitting Gap: {comparison_df.loc['Overfitting Gap', best_model_name]:.4f}")
print(f"  ROC AUC: {comparison_df.loc['ROC AUC', best_model_name]:.4f}")
print("="*80)

print("\nCreating Model Comparison Visualizations...")

# Create a figure with 2 subplots to show overfitting analysis
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Subplot 1: Test vs Train Accuracy Comparison
models_names = ['KNN', 'SVM', 'Random Forest']
train_acc = [train_metrics_knn['accuracy'], train_metrics_svm['accuracy'], train_metrics_rf['accuracy']]
test_acc = [test_metrics_knn['accuracy'], test_metrics_svm['accuracy'], test_metrics_rf['accuracy']]

x_pos = np.arange(len(models_names))
width = 0.35

bars1 = axes[0].bar(x_pos - width/2, train_acc, width, label='Train Accuracy', color='#4C72B0', alpha=0.8)
bars2 = axes[0].bar(x_pos + width/2, test_acc, width, label='Test Accuracy', color='#C44E52', alpha=0.8)

axes[0].set_title('Training vs Test Accuracy Comparison', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Accuracy', fontsize=12)
axes[0].set_ylim(0, 1.05)
axes[0].set_xticks(x_pos)
axes[0].set_xticklabels(models_names, rotation=0, fontsize=11)
axes[0].legend(loc='lower right', fontsize=10)
axes[0].grid(True, alpha=0.3, axis='y')

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)

# Subplot 2: Overfitting Gaps
overfitting_gaps = [overfitting_gap_knn, overfitting_gap_svm, overfitting_gap_rf]
colors_gap = ['green' if gap <= 0.05 else 'red' for gap in overfitting_gaps]
bars_gap = axes[1].bar(x_pos, overfitting_gaps, color=colors_gap, alpha=0.8)

# Add threshold line
axes[1].axhline(y=0.05, color='orange', linestyle='--', linewidth=2, label='Acceptable Threshold (0.05)')

axes[1].set_title('Overfitting Analysis (Train - Test Accuracy)', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Accuracy Gap', fontsize=12)
axes[1].set_xticks(x_pos)
axes[1].set_xticklabels(models_names, rotation=0, fontsize=11)
axes[1].legend(loc='upper left', fontsize=10)
axes[1].grid(True, alpha=0.3, axis='y')

# Add value labels
for bar in bars_gap:
    height = bar.get_height()
    axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.002,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "comparison_01_overfitting_analysis.png"), dpi=150, bbox_inches='tight')
plt.show()
plt.close()
print("   Created: comparison_01_overfitting_analysis.png")

# Second comparison: Test Metrics
fig, ax = plt.subplots(figsize=(10, 6))
metrics_df = comparison_df.iloc[3:7].T  # Test Precision, Recall, F1, Specificity
metrics_df.plot(kind="bar", ax=ax, colormap="viridis", width=0.8)
ax.set_title("Test Set Performance Metrics Comparison", fontsize=14, fontweight='bold')
ax.set_ylabel("Score", fontsize=12)
ax.set_xlabel("Model", fontsize=12)
ax.set_ylim(0, 1.05)
ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=11)
ax.legend(loc="lower right", fontsize=9, title="Metrics")
ax.grid(True, alpha=0.3, axis='y')
for container in ax.containers:
    ax.bar_label(container, fmt='%.3f', fontsize=7, rotation=90, padding=2)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "comparison_02_test_performance.png"), dpi=150, bbox_inches='tight')
plt.show()
plt.close()
print("   Created: comparison_02_test_performance.png")

# ROC Curves Comparison
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(fpr_knn, tpr_knn, label=f"KNN (AUC = {roc_auc_knn:.3f})", linewidth=2.5, color='#2E86AB')
ax.plot(fpr_svm, tpr_svm, label=f"SVM (AUC = {roc_auc_svm:.3f})", linewidth=2.5, color='#A23B72')
ax.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC = {roc_auc_rf:.3f})", linewidth=2.5, color='#F18F01')
ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Guess (AUC = 0.5)", linewidth=1.5, alpha=0.7)
ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=12)
ax.set_title("ROC Curves Comparison Across Models", fontsize=14, fontweight='bold')
ax.legend(loc="lower right", fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_facecolor('#f8f9fa')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "comparison_03_roc_curves.png"), dpi=150, bbox_inches='tight')
plt.show()
plt.close()
print("   Created: comparison_03_roc_curves.png")


# ==========================
# SAVE BEST MODEL & PREPROCESSORS
# ==========================
print("\n" + "="*60)
print("SAVING BEST MODEL AND PREPROCESSORS")
print("="*60)

# Save the best model
if best_model_name == 'KNN':
    best_model_to_save = knn_model
elif best_model_name == 'SVM':
    best_model_to_save = svm_model
else:
    best_model_to_save = rf_model

# Save the best model
pickle.dump(best_model_to_save, open("diabetes_model.pkl", "wb"))
print("Saved: diabetes_model.pkl (best model)")

# Save the imputer
pickle.dump(imputer, open("imputer.pkl", "wb"))
print("Saved: imputer.pkl")

# Save the fitted scaler
pickle.dump(scaler, open("scaler.pkl", "wb"))
print(" Saved: scaler.pkl (FITTED on training data)")

print(f"\n{'='*60}")
print(" All files saved successfully!")
print(f"  - diabetes_model.pkl: {best_model_name} (fitted)")
print(f"  - imputer.pkl: fitted on training data")
print(f"  - scaler.pkl: FITTED on training data (ready for prediction)")
print("="*60)


# ==========================
# FINAL SUMMARY
# ==========================
print("\n" + "="*80)
print("COMPLETE ANALYSIS SUMMARY")
print("="*80)

print("\nModel Performance Summary:")
print("-" * 60)
for model in comparison_df.columns:
    print(f"\n{model}:")
    print(f"  Train Accuracy:           {comparison_df.loc['Train Accuracy', model]:.4f}")
    print(f"  Test Accuracy:            {comparison_df.loc['Test Accuracy', model]:.4f}")
    print(f"  Overfitting Gap:          {comparison_df.loc['Overfitting Gap', model]:.4f}")
    print(f"  Test Precision:           {comparison_df.loc['Test Precision', model]:.4f}")
    print(f"  Test Recall (Sensitivity): {comparison_df.loc['Test Recall', model]:.4f}")
    print(f"  Test F1 Score:            {comparison_df.loc['Test F1 Score', model]:.4f}")
    print(f"  Specificity:              {comparison_df.loc['Specificity', model]:.4f}")
    print(f"  Negative Predictive Value: {comparison_df.loc['Negative Predictive Value', model]:.4f}")
    print(f"  ROC AUC:                  {comparison_df.loc['ROC AUC', model]:.4f}")

    # Overfitting indicator
    gap = comparison_df.loc['Overfitting Gap', model]
    if gap <= 0.03:
        status = " Excellent generalization"
    elif gap <= 0.05:
        status = " Good generalization"
    elif gap <= 0.08:
        status = "  Moderate overfitting detected"
    else:
        status = "  Significant overfitting - consider regularization"
    print(f"  Overfitting Status:       {status}")

print("\n" + "-" * 60)
print(f"\n BEST MODEL: {best_model_name}")
print(f"   Test Accuracy: {best_accuracy:.4f}")
print(f"   Train Accuracy: {comparison_df.loc['Train Accuracy', best_model_name]:.4f}")
print(f"   Overfitting Gap: {comparison_df.loc['Overfitting Gap', best_model_name]:.4f}")
print(f"   ROC AUC: {comparison_df.loc['ROC AUC', best_model_name]:.4f}")

print(f"\n{'='*80}")
print("SAVED FILES:")
print(f"   diabetes_model.pkl (Best model: {best_model_name})")
print(f"   imputer.pkl (fitted on training data)")
print(f"   scaler.pkl (FITTED - ready for prediction)")
print(f"  - knn_model.pkl")
print(f"  - svm_model.pkl")
print(f"  - rf_model.pkl")
print(f"\nAll {len(os.listdir(PLOT_DIR))} plots saved in the '{PLOT_DIR}/' folder.")
print("="*80)
print("\n Analysis complete! Models are ready for deployment.")
print("="*80)
