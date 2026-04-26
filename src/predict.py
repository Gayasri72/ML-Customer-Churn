"""
===============================================================
  PREDICTION SCRIPT — Telco Customer Churn
  Load a saved model and predict churn for new customer data
===============================================================

Usage:
    python predict.py                  # Interactive mode
    python predict.py --model rf       # Specify model (lr, dt, knn, rf)
"""

import pandas as pd
import numpy as np
import joblib
import argparse
import sys
from pathlib import Path

# ============================================================
# MODEL PATHS (relative to src/)
# ============================================================
MODEL_PATHS = {
    "lr": {
        "name": "Logistic Regression",
        "model": "notebook/logistic_regression/lr_churn_model.pkl",
        "features": "notebook/logistic_regression/lr_feature_columns.pkl"
    },
    "dt": {
        "name": "Decision Tree",
        "model": "notebook/decision_tree/churn_model.pkl",
        "features": "notebook/decision_tree/feature_columns.pkl"
    },
    "knn": {
        "name": "KNN",
        "model": "notebook/knn/knn_model.pkl",
        "features": None  # uses same features as others
    },
    "rf": {
        "name": "Random Forest",
        "model": "notebook/random_forest/rf_churn_model.pkl",
        "features": None
    }
}

# Feature columns from the processed dataset
FEATURE_COLUMNS = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
    'PhoneService', 'PaperlessBilling', 'MonthlyCharges', 'TotalCharges',
    'MultipleLines_No phone service', 'MultipleLines_Yes',
    'InternetService_Fiber optic', 'InternetService_No',
    'OnlineSecurity_No internet service', 'OnlineSecurity_Yes',
    'OnlineBackup_No internet service', 'OnlineBackup_Yes',
    'DeviceProtection_No internet service', 'DeviceProtection_Yes',
    'TechSupport_No internet service', 'TechSupport_Yes',
    'StreamingTV_No internet service', 'StreamingTV_Yes',
    'StreamingMovies_No internet service', 'StreamingMovies_Yes',
    'Contract_One year', 'Contract_Two year',
    'PaymentMethod_Credit card (automatic)',
    'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check'
]


def load_model(model_key):
    """Load a saved model by key (lr, dt, knn, rf)."""
    if model_key not in MODEL_PATHS:
        print(f"Error: Unknown model '{model_key}'. Choose from: {list(MODEL_PATHS.keys())}")
        sys.exit(1)

    config = MODEL_PATHS[model_key]
    model_path = Path(__file__).parent / config["model"]

    if not model_path.exists():
        print(f"Error: Model file not found at {model_path}")
        print("Please train the model first by running the corresponding notebook.")
        sys.exit(1)

    model = joblib.load(model_path)
    print(f"Loaded model: {config['name']}")
    return model


def get_sample_customer():
    """Return a sample customer for demonstration."""
    return {
        'gender': 1,                             # Male=1, Female=0
        'SeniorCitizen': 0,                      # 0=No, 1=Yes
        'Partner': 0,                            # 0=No, 1=Yes
        'Dependents': 0,                         # 0=No, 1=Yes
        'tenure': 2,                             # months
        'PhoneService': 1,                       # 0=No, 1=Yes
        'PaperlessBilling': 1,                   # 0=No, 1=Yes
        'MonthlyCharges': 70.70,
        'TotalCharges': 151.65,
        'MultipleLines_No phone service': 0,
        'MultipleLines_Yes': 0,
        'InternetService_Fiber optic': 1,
        'InternetService_No': 0,
        'OnlineSecurity_No internet service': 0,
        'OnlineSecurity_Yes': 0,
        'OnlineBackup_No internet service': 0,
        'OnlineBackup_Yes': 0,
        'DeviceProtection_No internet service': 0,
        'DeviceProtection_Yes': 0,
        'TechSupport_No internet service': 0,
        'TechSupport_Yes': 0,
        'StreamingTV_No internet service': 0,
        'StreamingTV_Yes': 0,
        'StreamingMovies_No internet service': 0,
        'StreamingMovies_Yes': 0,
        'Contract_One year': 0,
        'Contract_Two year': 0,
        'PaymentMethod_Credit card (automatic)': 0,
        'PaymentMethod_Electronic check': 1,
        'PaymentMethod_Mailed check': 0
    }


def predict_churn(model, customer_data):
    """Predict churn for a single customer."""
    # Create DataFrame with correct feature order
    df = pd.DataFrame([customer_data], columns=FEATURE_COLUMNS)

    # Note: The model was trained on StandardScaler-transformed data.
    # For accurate predictions on new raw data, you would need to apply
    # the same scaler. This demo uses pre-scaled values.

    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0]

    return prediction, probability


def main():
    parser = argparse.ArgumentParser(description="Predict customer churn")
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='rf',
        choices=['lr', 'dt', 'knn', 'rf'],
        help='Model to use: lr (Logistic Regression), dt (Decision Tree), knn (KNN), rf (Random Forest)'
    )
    args = parser.parse_args()

    print("=" * 55)
    print("  CUSTOMER CHURN PREDICTION")
    print("=" * 55)

    # Load model
    model = load_model(args.model)

    # Use sample customer
    customer = get_sample_customer()

    print("\n--- Customer Profile ---")
    print(f"  Tenure:          {customer['tenure']} months")
    print(f"  Monthly Charges:  ${customer['MonthlyCharges']}")
    print(f"  Total Charges:    ${customer['TotalCharges']}")
    print(f"  Contract:         Month-to-month")
    print(f"  Internet:         Fiber optic")
    print(f"  Payment:          Electronic check")

    # Predict
    prediction, probability = predict_churn(model, customer)

    print("\n--- Prediction ---")
    if prediction == 1:
        print(f"  ⚠️  CHURN RISK: YES")
    else:
        print(f"  ✅  CHURN RISK: NO")

    print(f"  Probability of churn:     {probability[1]:.2%}")
    print(f"  Probability of no churn:  {probability[0]:.2%}")

    print("\n" + "=" * 55)
    print(f"  Model used: {MODEL_PATHS[args.model]['name']}")
    print("=" * 55)


if __name__ == "__main__":
    main()
