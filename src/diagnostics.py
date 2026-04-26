"""
===============================================================
  THRESHOLD OPTIMIZATION & LEARNING CURVES
  Diagnostic tools for the Telco Customer Churn models
===============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from imblearn.over_sampling import SMOTE

sns.set_style('whitegrid')

# ============================================================
# 1. LOAD DATA & PREPARE
# ============================================================
print("=" * 60)
print("  Loading data...")
print("=" * 60)

df = pd.read_csv("../data/processed/churn_processed.csv")

X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

print(f"Train (SMOTE): {X_train_sm.shape[0]} | Test: {X_test.shape[0]}")

# ============================================================
# 2. THRESHOLD OPTIMIZATION
# ============================================================
print(f"\n{'=' * 60}")
print("  THRESHOLD OPTIMIZATION")
print("=" * 60)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=7, weights='distance'),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)
}

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

best_thresholds = {}

for idx, (name, model) in enumerate(models.items()):
    model.fit(X_train_sm, y_train_sm)
    y_prob = model.predict_proba(X_test)[:, 1]

    thresholds = np.arange(0.1, 0.91, 0.01)
    f1_scores = []
    precision_scores = []
    recall_scores = []

    for t in thresholds:
        y_pred_t = (y_prob >= t).astype(int)
        f1_scores.append(f1_score(y_test, y_pred_t))
        precision_scores.append(precision_score(y_test, y_pred_t, zero_division=0))
        recall_scores.append(recall_score(y_test, y_pred_t))

    best_idx = np.argmax(f1_scores)
    best_t = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    best_thresholds[name] = best_t

    # Youden's J statistic (sensitivity + specificity - 1)
    youden_j = [r - (1 - p) for r, p in zip(recall_scores, precision_scores)]
    best_youden_idx = np.argmax(youden_j)

    ax = axes[idx]
    ax.plot(thresholds, f1_scores, color='#2196F3', lw=2, label='F1 Score')
    ax.plot(thresholds, precision_scores, color='#4CAF50', lw=1.5, linestyle='--', label='Precision')
    ax.plot(thresholds, recall_scores, color='#FF9800', lw=1.5, linestyle='--', label='Recall')
    ax.axvline(x=best_t, color='red', lw=1.5, linestyle=':', label=f'Best F1 threshold={best_t:.2f}')
    ax.axvline(x=0.5, color='gray', lw=1, linestyle='--', alpha=0.5, label='Default (0.5)')
    ax.set_xlabel('Threshold', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(f'{name}\nBest F1={best_f1:.4f} @ t={best_t:.2f}', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='lower left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.1, 0.9)

    print(f"\n  {name}:")
    print(f"    Default (t=0.50) → F1={f1_scores[40]:.4f}")
    print(f"    Optimal (t={best_t:.2f}) → F1={best_f1:.4f}")
    print(f"    Youden's J best  → t={thresholds[best_youden_idx]:.2f}")

plt.suptitle('Threshold Optimization — All Models', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig("../data/processed/threshold_optimization.png", dpi=150, bbox_inches='tight')
plt.show()

print("\nBest thresholds saved to: data/processed/threshold_optimization.png")

# ============================================================
# 3. LEARNING CURVES
# ============================================================
print(f"\n{'=' * 60}")
print("  LEARNING CURVES")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for idx, (name, model) in enumerate(models.items()):
    print(f"  Computing learning curve for {name}...")

    train_sizes, train_scores, val_scores = learning_curve(
        model, X_train_sm, y_train_sm,
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=5,
        scoring='f1',
        n_jobs=-1,
        random_state=42
    )

    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    ax = axes[idx]
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='#2196F3')
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='#FF9800')
    ax.plot(train_sizes, train_mean, 'o-', color='#2196F3', lw=2, label=f'Training (final={train_mean[-1]:.3f})')
    ax.plot(train_sizes, val_mean, 'o-', color='#FF9800', lw=2, label=f'Validation (final={val_mean[-1]:.3f})')

    gap = train_mean[-1] - val_mean[-1]
    if gap > 0.05:
        diagnosis = "⚠️ Overfitting"
    elif val_mean[-1] < 0.7:
        diagnosis = "⚠️ Underfitting"
    else:
        diagnosis = "✅ Good fit"

    ax.set_xlabel('Training Set Size', fontsize=12)
    ax.set_ylabel('F1 Score', fontsize=12)
    ax.set_title(f'{name}\n{diagnosis} (gap={gap:.3f})', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.5, 1.05)

plt.suptitle('Learning Curves — Bias/Variance Diagnosis', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig("../data/processed/learning_curves.png", dpi=150, bbox_inches='tight')
plt.show()

print("\nLearning curves saved to: data/processed/learning_curves.png")

# ============================================================
# 4. SUMMARY
# ============================================================
print(f"\n{'=' * 60}")
print("  SUMMARY — OPTIMAL THRESHOLDS")
print("=" * 60)

for name, t in best_thresholds.items():
    print(f"  {name:25s} → t={t:.2f}")

print("=" * 60)
print("  DIAGNOSTICS COMPLETE!")
print("=" * 60)
