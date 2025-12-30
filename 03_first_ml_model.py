import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

# 1️⃣ Load data
features = pd.read_csv("data/elliptic_txs_features.csv", header=None)
classes = pd.read_csv("data/elliptic_txs_classes.csv", header=None)

# 2️⃣ Rename columns
features = features.rename(columns={0: "txId"})
classes = classes.rename(columns={0: "txId", 1: "class"})

# 3️⃣ Fix datatype mismatch
features["txId"] = features["txId"].astype(str)
classes["txId"] = classes["txId"].astype(str)

# 4️⃣ Merge
data = features.merge(classes, on="txId")

# 5️⃣ Separate features and label
X = data.drop(columns=["txId", "class"])
y = data["class"]

print("X shape:", X.shape)
print("y shape:", y.shape)

# 6️⃣ Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# 7️⃣ Train Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 8️⃣ Predictions
y_pred = model.predict(X_test)

# 9️⃣ Evaluation
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
