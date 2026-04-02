"""
===============================================================
  MODEL COMPARISON — Telco Customer Churn Prediction
  Compares: Logistic Regression, Decision Tree, KNN, Random Forest
  With: OneHotEncoding, SMOTE, 5-Fold Cross-Validation, GridSearchCV
===============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix,
    roc_curve, roc_auc_score
)
from imblearn.over_sampling import SMOTE

# ============================================================
# 1. LOAD & PREPROCESS (with OneHotEncoding fix)
# ============================================================
print("=" * 60)
print("  STEP 1: Loading & Preprocessing Dataset")
print("=" * 60)

df = pd.read_csv("../data/raw/churn.csv")
print(f"Raw shape: {df.shape}")

# Drop customerID
df.drop("customerID", axis=1, inplace=True)

# Fix TotalCharges
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.dropna(inplace=True)

# Encode target
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# Binary columns -> LabelEncoder (correct for 2-value features)
# Multi-class columns -> OneHotEncoding (correct for nominal features)
binary_cols = []
multi_cols = []
for col in df.select_dtypes(include="object").columns:
    if df[col].nunique() == 2:
        binary_cols.append(col)
    else:
        multi_cols.append(col)

for col in binary_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

df = pd.get_dummies(df, columns=multi_cols, drop_first=True)

# Cast bool to int
bool_cols = df.select_dtypes(include="bool").columns
df[bool_cols] = df[bool_cols].astype(int)

print(f"After encoding: {df.shape[1]} features")
print(f"Class distribution: {df['Churn'].value_counts().to_dict()}")

# Separate features & target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Scale features
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# ============================================================
# 2. TRAIN/TEST SPLIT + SMOTE
# ============================================================
print(f"\n{'=' * 60}")
print("  STEP 2: Train/Test Split + SMOTE")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

print(f"Train (original):  {X_train.shape[0]} samples")
print(f"Train (SMOTE):     {X_train_sm.shape[0]} samples")
print(f"Test:              {X_test.shape[0]} samples")
print(f"SMOTE class dist:  {pd.Series(y_train_sm).value_counts().to_dict()}")

# ============================================================
# 3. DEFINE MODELS & HYPERPARAMETER GRIDS
# ============================================================
models = {
    "Logistic Regression": {
        "estimator": LogisticRegression(max_iter=1000, random_state=42),
        "params": {
            "C": [0.01, 0.1, 1, 10, 100],
            "solver": ["lbfgs", "liblinear"],
            "penalty": ["l2"]
        }
    },
    "Decision Tree": {
        "estimator": DecisionTreeClassifier(random_state=42),
        "params": {
            "max_depth": [5, 10, 15, 20],
            "min_samples_split": [2, 5, 10, 20],
            "min_samples_leaf": [1, 2, 5, 10],
            "criterion": ["gini", "entropy"]
        }
    },
    "KNN": {
        "estimator": KNeighborsClassifier(),
        "params": {
            "n_neighbors": list(range(3, 21)),
            "weights": ["uniform", "distance"],
            "metric": ["euclidean", "manhattan"]
        }
    },
    "Random Forest": {
        "estimator": RandomForestClassifier(random_state=42, n_jobs=-1),
        "params": {
            "n_estimators": [100, 200, 300],
            "max_depth": [10, 20, 30, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4]
        }
    }
}

# ============================================================
# 4. TRAIN, TUNE & EVALUATE EACH MODEL
# ============================================================
results = []

for name, config in models.items():
    print(f"\n{'=' * 60}")
    print(f"  TRAINING: {name}")
    print("=" * 60)

    # --- GridSearchCV on SMOTE-balanced training data ---
    grid = GridSearchCV(
        config["estimator"],
        config["params"],
        cv=5,
        scoring="f1",
        n_jobs=-1,
        verbose=0
    )
    grid.fit(X_train_sm, y_train_sm)
    best_model = grid.best_estimator_

    print(f"  Best params: {grid.best_params_}")

    # --- 5-Fold Cross-Validation ---
    cv_scores = cross_val_score(best_model, X_train_sm, y_train_sm, cv=5, scoring="f1")
    print(f"  CV F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # --- Evaluate on unseen test data ---
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    acc   = accuracy_score(y_test, y_pred)
    prec  = precision_score(y_test, y_pred)
    rec   = recall_score(y_test, y_pred)
    f1    = f1_score(y_test, y_pred)
    auc   = roc_auc_score(y_test, y_prob)

    print(f"  Test Accuracy:  {acc:.4f}")
    print(f"  Test Precision: {prec:.4f}")
    print(f"  Test Recall:    {rec:.4f}  ← (key for catching churners)")
    print(f"  Test F1 Score:  {f1:.4f}")
    print(f"  Test AUC-ROC:   {auc:.4f}")

    results.append({
        "Model": name,
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1 Score": round(f1, 4),
        "AUC-ROC": round(auc, 4),
        "CV F1 (mean)": round(cv_scores.mean(), 4),
        "CV F1 (std)": round(cv_scores.std(), 4),
        "Best Params": str(grid.best_params_),
        "_y_prob": y_prob  # for ROC plot later
    })

# ============================================================
# 5. COMPARISON TABLE
# ============================================================
print(f"\n{'=' * 60}")
print("  FINAL MODEL COMPARISON")
print("=" * 60)

results_df = pd.DataFrame(results).drop(columns=["_y_prob", "Best Params"])
results_df = results_df.sort_values("F1 Score", ascending=False).reset_index(drop=True)
results_df.index = results_df.index + 1  # rank starting from 1
results_df.index.name = "Rank"

print(results_df.to_string())

# --- Determine best model ---
best = results_df.iloc[0]
print(f"\n🏆 BEST MODEL: {best['Model']}")
print(f"   F1 Score: {best['F1 Score']:.4f}  |  Recall: {best['Recall']:.4f}  |  AUC: {best['AUC-ROC']:.4f}")

# ============================================================
# 6. VISUALIZATIONS
# ============================================================

# --- 6a. Bar chart comparison ---
metrics = ["Accuracy", "Precision", "Recall", "F1 Score", "AUC-ROC"]
fig, axes = plt.subplots(1, len(metrics), figsize=(20, 5))

for i, metric in enumerate(metrics):
    data = pd.DataFrame(results)[["Model", metric]].sort_values(metric, ascending=True)
    colors = ["#FF7043" if m != best["Model"] else "#4CAF50" for m in data["Model"]]
    axes[i].barh(data["Model"], data[metric], color=colors, edgecolor="black", linewidth=0.5)
    axes[i].set_title(metric, fontsize=13, fontweight="bold")
    axes[i].set_xlim(0, 1.05)
    for j, v in enumerate(data[metric]):
        axes[i].text(v + 0.01, j, f"{v:.3f}", va="center", fontsize=10)

plt.suptitle("Model Comparison — All Metrics", fontsize=16, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("../data/processed/model_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
print("Chart saved to: data/processed/model_comparison.png")

# --- 6b. ROC curves (all models on one plot) ---
plt.figure(figsize=(8, 6))
colors = ["#2196F3", "#FF9800", "#9C27B0", "#4CAF50"]

for i, r in enumerate(results):
    fpr, tpr, _ = roc_curve(y_test, r["_y_prob"])
    plt.plot(fpr, tpr, color=colors[i], lw=2, label=f'{r["Model"]} (AUC={r["AUC-ROC"]:.3f})')

plt.plot([0, 1], [0, 1], "--", color="gray", lw=1)
plt.xlabel("False Positive Rate", fontsize=13)
plt.ylabel("True Positive Rate", fontsize=13)
plt.title("ROC Curves — All Models", fontsize=15, fontweight="bold")
plt.legend(loc="lower right", fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("../data/processed/roc_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
print("ROC chart saved to: data/processed/roc_comparison.png")

print(f"\n{'=' * 60}")
print("  COMPARISON COMPLETE!")
print("=" * 60)
