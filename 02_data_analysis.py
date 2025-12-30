import os
import pandas as pd

# Load CSVs (no headers in Elliptic dataset)
features = pd.read_csv("data/elliptic_txs_features.csv", header=None)
classes = pd.read_csv("data/elliptic_txs_classes.csv", header=None)

# Rename columns
features = features.rename(columns={0: "txId"})
classes = classes.rename(columns={0: "txId", 1: "class"})

# 🔴 FIX: make txId SAME TYPE in both
features["txId"] = features["txId"].astype(str)
classes["txId"] = classes["txId"].astype(str)

print("Features txId dtype:", features["txId"].dtype)
print("Classes txId dtype:", classes["txId"].dtype)

# Merge
data = features.merge(classes, on="txId")

print("Total transactions:", data.shape[0])
print(data.head())
import matplotlib.pyplot as plt

# Count fraud vs non-fraud
fraud_counts = data["class"].value_counts()

print("\nFraud vs Non-Fraud counts:")
print(fraud_counts)

# Plot distribution
plt.figure(figsize=(6,4))
fraud_counts.plot(kind="bar")
plt.title("Fraud vs Non-Fraud Transaction Count")
plt.xlabel("Class (1 = Fraud, 2 = Non-Fraud)")
plt.ylabel("Number of Transactions")
plt.tight_layout()
plt.show()
