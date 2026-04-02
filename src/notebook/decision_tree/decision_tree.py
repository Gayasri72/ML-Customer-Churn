#!/usr/bin/env python
# coding: utf-8

# In[2]:


# =====================
# 1. IMPORT LIBRARIES
# =====================
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)

import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import subprocess
from imblearn.over_sampling import SMOTE

print("Libraries loaded")


# In[3]:


#==============================
# 2. RUN EXISTING PREPROCESS.PY
#==============================
subprocess.run(["python", "preprocess.py"])
print("Preprocessing completed")


# In[5]:


# =====================
# 3. LOAD DATASET
# =====================
file_path = Path().resolve().parent.parent.parent / "data" / "processed" / "churn_processed.csv"
df = pd.read_csv(file_path)

print("Shape:", df.shape)
df.head()


# In[6]:


# =============
# 4. SPLIT DATA
# =============

X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train:", X_train.shape)
print("Test:", X_test.shape)
print(y_train.value_counts())



# In[7]:


# ======================
# 5. DECISION TREE MODEL
# ======================
model = DecisionTreeClassifier(
    max_depth=6,
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=42
)

model.fit(X_train, y_train)

print("Model trained")

#==========
#PREDICTION
#==========

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]


# In[8]:


# =============
# 6. EVALUATION
# =============
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))


# In[9]:


# =====================
# 7. CONFUSION MATRIX
# =====================
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# In[10]:


# =====================
# 8. FEATURE IMPORTANCE
# =====================

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

importance.head(10)

# Plot feature importance
plt.figure(figsize=(10,6))
sns.barplot(x="Importance", y="Feature", data=importance)
plt.title("Feature Importance")
plt.show()


# In[11]:


# ==============================
# 9. DECISION TREE VISUALIZATION
# ==============================
plt.figure(figsize=(25, 12))

plot_tree(
    model,                     # or model if you didn't use GridSearch
    feature_names=X.columns,
    class_names=["No Churn", "Churn"],
)

plt.title("Decision Tree Visualization")
plt.show()


# In[12]:


# ===================
# 10. CROSS VALIDATION
# ===================

scores = cross_val_score(model, X, y, cv=5)

print("CV Scores:", scores)
print("Mean CV F1 Score:", scores.mean())


# In[13]:


# ======================================
# 11.HYPERPARAMETER TUNING (GRID SEARCH)
# ======================================

param_grid = {
    "max_depth": [5, 10, 15, 20, 30],
    "min_samples_split": [2, 5, 10, 20],
    "min_samples_leaf": [1, 2, 5, 10],
    "criterion": ["gini", "entropy"]
}

grid = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring="f1"
)

grid.fit(X_train, y_train)

print("Best Parameters:", grid.best_params_)


# In[14]:


# ==============
# 12. BEST MODEL
# ==============

best_model = grid.best_estimator_


# In[15]:


# =====================
# 13. OVERFITTING CHECK
# =====================

train_acc = best_model.score(X_train, y_train)
test_acc = best_model.score(X_test, y_test)

print("Train Accuracy:", train_acc)
print("Test Accuracy:", test_acc)
print("Gap:", train_acc - test_acc)


# In[16]:


# =========================
# 14. TUNED MODEL EVALUATION
# =========================

y_pred2 = best_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred2))
print("Precision:", precision_score(y_test, y_pred2))
print("Recall:", recall_score(y_test, y_pred2))
print("F1 Score:", f1_score(y_test, y_pred2))


print("\nClassification Report:\n")
print(classification_report(y_test, y_pred2))


# In[17]:


#=============================
# 15. CONFUSION MATRIX (TUNED)
# ============================

cm2 = confusion_matrix(y_test, y_pred2)

plt.figure(figsize=(6,4))
sns.heatmap(cm2, annot=True, fmt="d", cmap="Greens")
plt.title("Tuned Model Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# In[18]:


#==============
# 16. ROC CURVE
# =============

from sklearn.metrics import roc_curve, auc

y_prob = best_model.predict_proba(X_test)[:,1]

fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0,1],[0,1],'--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()


# In[20]:


# =====================
# 17. MODEL COMPARISON
# =====================
comparison = pd.DataFrame({
    "Model": ["Decision Tree", "Tuned Decision Tree"],
    "Accuracy": [
        accuracy_score(y_test, y_pred),
        accuracy_score(y_test, y_pred2)
    ],
    "F1 Score": [
        f1_score(y_test, y_pred),
        f1_score(y_test, y_pred2)
    ]
})

comparison


# In[24]:


#===============
# 18. SAVE MODEL
# ==============

joblib.dump(best_model, "churn_model.pkl")
joblib.dump(X.columns, "feature_columns.pkl")

print("Model saved successfully")


# In[25]:


# =================
# 19. FINAL SUMMARY
# =================
print("Decision Tree Model Completed Successfully")

