#!/usr/bin/env python
# coding: utf-8

# # Logistic Regression — Telco Customer Churn Prediction
# 
# **Dataset:** [Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)  
# **Algorithm:** Logistic Regression (Supervised Learning)  
# **Goal:** Predict whether a customer will churn (Yes/No) based on service usage and account details.

# ## 1. Import Libraries

# In[24]:


import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
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


# ## 2. Run Preprocessing
# 
# The shared `preprocess.py` script handles:
# - Dropping `customerID`
# - Fixing `TotalCharges` data type
# - Handling missing values
# - Label encoding categorical columns
# - Standard scaling all features

# In[25]:


subprocess.run(["python", "preprocess.py"])
print("Preprocessing completed")


# ## 3. Load Processed Data

# In[26]:


df = pd.read_csv("../../../data/processed/churn_processed.csv")

print("Shape:", df.shape)
df.head()


# ## 4. Separate Features & Target

# In[27]:


X = df.drop("Churn", axis=1)
y = df["Churn"]

print("Feature columns:", list(X.columns))
print("\nTarget distribution:")
print(y.value_counts())


# ## 5. Train / Test Split
# 
# > `stratify=y` ensures both train and test sets maintain the same churn ratio.

# In[28]:


X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train size:", X_train.shape)
print("Test size :", X_test.shape)


# ## 6. Handle Class Imbalance with SMOTE
# 
# The Telco dataset is imbalanced — churned customers are a minority class.  
# **SMOTE** (Synthetic Minority Oversampling Technique) generates synthetic samples for the minority class to balance the training set.

# In[29]:


smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

print("After SMOTE:")
print(pd.Series(y_train_sm).value_counts())


# ## 7. Baseline Logistic Regression Model
# 
# **Logistic Regression** is a linear classification algorithm that models the probability of a binary outcome.  
# It uses the sigmoid function to output values between 0 and 1, making it well-suited for binary classification tasks like churn prediction.

# In[30]:


lr_model = LogisticRegression(
    max_iter=1000,
    random_state=42,
    solver='lbfgs'
)

lr_model.fit(X_train_sm, y_train_sm)
print("Baseline model training complete!")


# ## 8. Baseline Model Predictions & Evaluation

# In[31]:


y_pred = lr_model.predict(X_test)

print("Baseline Logistic Regression Results:")
print(f"  Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
print(f"  Precision : {precision_score(y_test, y_pred):.4f}")
print(f"  Recall    : {recall_score(y_test, y_pred):.4f}")
print(f"  F1 Score  : {f1_score(y_test, y_pred):.4f}")


# ## 9. Classification Report

# In[32]:


print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))


# ## 10. Confusion Matrix — Baseline

# In[33]:


cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["No Churn", "Churn"],
            yticklabels=["No Churn", "Churn"])
plt.title("Logistic Regression - Confusion Matrix (Baseline)")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.show()


# ## 11. Cross Validation
# 
# 5-fold cross validation is used to verify the model does not overfit on the training set.

# In[34]:


cv_scores = cross_val_score(lr_model, X_train_sm, y_train_sm, cv=5, scoring='accuracy')

print(f"Cross Validation Scores : {cv_scores}")
print(f"Mean CV Accuracy        : {cv_scores.mean():.4f}")
print(f"Standard Deviation      : {cv_scores.std():.4f}")


# ## 12. Hyperparameter Tuning with GridSearchCV
# 
# Key hyperparameters tuned:
# - **C** — Regularization strength (smaller = stronger regularization)
# - **solver** — Optimization algorithm used to fit the model

# In[35]:


param_grid = {
    'C'      : [0.01, 0.1, 1, 10, 100],
    'solver' : ['lbfgs', 'liblinear'],
    'penalty': ['l2']
}

grid_search = GridSearchCV(
    LogisticRegression(max_iter=1000, random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train_sm, y_train_sm)

print("Best Parameters:", grid_search.best_params_)
print("Best CV Score  :", round(grid_search.best_score_, 4))


# ## 13. Tuned Model Predictions & Evaluation

# In[36]:


best_model = grid_search.best_estimator_
y_pred2 = best_model.predict(X_test)

print("Tuned Logistic Regression Results:")
print(f"  Accuracy  : {accuracy_score(y_test, y_pred2):.4f}")
print(f"  Precision : {precision_score(y_test, y_pred2):.4f}")
print(f"  Recall    : {recall_score(y_test, y_pred2):.4f}")
print(f"  F1 Score  : {f1_score(y_test, y_pred2):.4f}")


# ## 14. Confusion Matrix — Tuned Model

# In[38]:


cm2 = confusion_matrix(y_test, y_pred2)

plt.figure(figsize=(6, 4))
sns.heatmap(cm2, annot=True, fmt='d', cmap='Greens',
            xticklabels=["No Churn", "Churn"],
            yticklabels=["No Churn", "Churn"])
plt.title("Logistic Regression - Confusion Matrix (Tuned)")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.show()


# ## 15. Feature Importance (Coefficients)
# 
# In Logistic Regression, the magnitude of each coefficient indicates how strongly that feature influences the prediction.  
# Positive coefficients increase churn probability; negative coefficients decrease it.

# In[40]:


coef_df = pd.DataFrame({
    'Feature'    : X.columns,
    'Coefficient': best_model.coef_[0]
}).sort_values('Coefficient', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=coef_df, x='Coefficient', y='Feature', palette='coolwarm')
plt.title("Logistic Regression - Feature Coefficients")
plt.tight_layout()
plt.show()


# ## 16. ROC Curve
# 
# The ROC curve shows the trade-off between **True Positive Rate** and **False Positive Rate** at different classification thresholds.  
# **AUC (Area Under Curve)** closer to 1.0 means better model performance.

# In[42]:


from sklearn.metrics import roc_curve, auc

y_prob = best_model.predict_proba(X_test)[:, 1]

fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}", color='blue')
plt.plot([0, 1], [0, 1], '--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Logistic Regression")
plt.legend()
plt.tight_layout()
plt.show()


# ## 17. Model Comparison
# 
# Comparing Baseline vs Tuned Logistic Regression performance.

# In[43]:


comparison = pd.DataFrame({
    "Model": ["Logistic Regression", "Tuned Logistic Regression"],
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


# ## 18. Save Model

# In[44]:


joblib.dump(best_model, "lr_churn_model.pkl")
joblib.dump(X.columns, "lr_feature_columns.pkl")

print("Model saved successfully")


# ## 19. Final Summary

# In[45]:


print("Logistic Regression Model Completed Successfully")

