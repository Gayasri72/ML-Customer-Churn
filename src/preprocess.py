import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# =========================
# 1. Load Dataset
# =========================
print("Loading dataset...")
df = pd.read_csv("data/raw/churn.csv")

print("Initial shape:", df.shape)

# =========================
# 2. Drop unnecessary columns
# =========================
if "customerID" in df.columns:
    df.drop("customerID", axis=1, inplace=True)

# =========================
# 3. Fix data types
# =========================
# Convert TotalCharges to numeric (it may contain spaces)
if "TotalCharges" in df.columns:
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# =========================
# 4. Handle missing values
# =========================
missing_before = df.isnull().sum().sum()
print("Missing values before:", missing_before)

df.dropna(inplace=True)

missing_after = df.isnull().sum().sum()
print("Missing values after:", missing_after)

# =========================
# 5. Encode categorical variables
# =========================
print("Encoding categorical features...")

# --- 5a. Encode the target column (binary: Yes/No) ---
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# --- 5b. Identify binary vs multi-class categorical columns ---
binary_cols = []
multi_cols = []

for col in df.select_dtypes(include="object").columns:
    n_unique = df[col].nunique()
    if n_unique == 2:
        binary_cols.append(col)
    else:
        multi_cols.append(col)

print(f"  Binary columns  ({len(binary_cols)}): {binary_cols}")
print(f"  Multi-class cols ({len(multi_cols)}): {multi_cols}")

# --- 5c. Label-encode binary columns (0/1 is correct for 2-value features) ---
for col in binary_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# --- 5d. One-Hot Encode multi-class nominal columns ---
# drop_first=True avoids the dummy-variable trap
df = pd.get_dummies(df, columns=multi_cols, drop_first=True)

# Ensure all dummy columns are int (not bool) for compatibility with StandardScaler
bool_cols = df.select_dtypes(include="bool").columns
df[bool_cols] = df[bool_cols].astype(int)

# =========================
# 6. Separate features & target
# =========================
if "Churn" not in df.columns:
    raise Exception("Target column 'Churn' not found!")

X = df.drop("Churn", axis=1)
y = df["Churn"]

# =========================
# 7. Feature Scaling
# =========================
print("Scaling features...")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =========================
# 8. Save processed dataset
# =========================
processed_df = pd.DataFrame(X_scaled, columns=X.columns)
processed_df["Churn"] = y.values

output_path = "data/processed/churn_processed.csv"
processed_df.to_csv(output_path, index=False)

print("Processed dataset saved to:", output_path)
print("Final shape:", processed_df.shape)

# =========================
# 9. Done
# =========================
print("Preprocessing completed successfully!")