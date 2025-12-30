import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load data
features = pd.read_csv("data/elliptic_txs_features.csv", header=None)
classes = pd.read_csv("data/elliptic_txs_classes.csv", header=None)

# Rename columns
features = features.rename(columns={0: "txId"})
classes = classes.rename(columns={0: "txId", 1: "class"})

# Fix datatypes
features["txId"] = features["txId"].astype(str)
classes["txId"] = classes["txId"].astype(str)

# Merge
data = features.merge(classes, on="txId")

# Separate X and y
X = data.drop(columns=["txId", "class"])
y = data["class"]

# Train-test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# ============================
# 1️⃣ Logistic Regression (Balanced)
# ============================
log_model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)
log_model.fit(X_train, y_train)

log_pred = log_model.predict(X_test)

print("\nLOGISTIC REGRESSION (BALANCED)")
print(confusion_matrix(y_test, log_pred))
print(classification_report(y_test, log_pred))

# ============================
# 2️⃣ Random Forest
# ============================
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)
rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

print("\nRANDOM FOREST")
print(confusion_matrix(y_test, rf_pred))
print(classification_report(y_test, rf_pred))

import numpy as np

# Feature importance from Random Forest
importances = rf_model.feature_importances_

# Create a DataFrame for better readability
feature_importance_df = pd.DataFrame({
    "feature_index": X.columns,
    "importance": importances
}).sort_values(by="importance", ascending=False)

print("\nTop 10 Important Features:")
print(feature_importance_df.head(10))

import joblib

# Save the trained model
joblib.dump(rf_model, "random_forest_fraud_model.pkl")

print("\nModel saved as random_forest_fraud_model.pkl")
