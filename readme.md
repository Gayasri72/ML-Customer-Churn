# ML Customer Churn Prediction

## 📌 Project Overview

This project predicts customer churn using supervised machine learning models.

## 📊 Dataset

Telco Customer Churn dataset (Kaggle)

## ⚙️ Models Used

* Logistic Regression
* Decision Tree
* KNN
* Random Forest

## 📈 Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1 Score
* AUC-ROC

## 👨‍👩‍👧‍👦 Team Responsibilities

* Preprocessing: Gayasri, Pethum
* Logistic Regression: (Team Member)
* Decision Tree: Samidi
* KNN: Nethmina
* Random Forest: Gayasri
 

## 🚀 How to Run

1. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

2. Run preprocessing:
   ```bash
   python src/preprocess.py
   ```

3. Train individual models:
   Use scripts inside `src/notebook/`

4. Compare all models:
   ```bash
   cd src
   python compare_models.py
   ```

## 📌 Notes

* Same dataset and preprocessing used for all models
* SMOTE used to handle class imbalance (applied only on training data)
* GridSearchCV with 5-fold cross-validation used for hyperparameter tuning
* All models optimized for F1 score to ensure fair comparison
