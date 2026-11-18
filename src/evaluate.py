
import torch
import joblib
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score
)

from src.preprocessing import load_data, preprocess_data
from src.model import CreditRiskNN


# Load dataset
df = load_data("data/Loan_default.csv")

X, y = preprocess_data(df)


# Recreate exactly the same stratified splits as train.py
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)


# Load scaler and feature columns
scaler = joblib.load("models/scaler.pkl")

feature_columns = joblib.load(
    "models/feature_columns.pkl"
)

X_val = X_val[feature_columns]
X_test = X_test[feature_columns]


# Scale validation and testing data
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)


# Convert data to tensors
X_val_tensor = torch.tensor(
    X_val,
    dtype=torch.float32
)

X_test_tensor = torch.tensor(
    X_test,
    dtype=torch.float32
)


# Initialize and load model
model = CreditRiskNN(
    input_size=len(feature_columns)
)

model.load_state_dict(
    torch.load(
        "models/credit_risk_model.pth",
        map_location="cpu",
        weights_only=True
    )
)

model.eval()


# Generate validation and test probabilities
with torch.no_grad():

    val_logits = model(X_val_tensor)

    val_probabilities = torch.sigmoid(
        val_logits
    ).numpy().flatten()

    test_logits = model(X_test_tensor)

    test_probabilities = torch.sigmoid(
        test_logits
    ).numpy().flatten()


# ------------------------------------------------
# THRESHOLD OPTIMIZATION USING VALIDATION DATA
# ------------------------------------------------

thresholds = np.arange(0.20, 0.81, 0.01)

best_threshold = 0.50
best_f1 = 0

print("\nVALIDATION RESULTS")

for threshold in thresholds:

    val_predictions = (
        val_probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        y_val,
        val_predictions,
        zero_division=0
    )

    recall = recall_score(
        y_val,
        val_predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_val,
        val_predictions,
        zero_division=0
    )

    print(
        f"Threshold: {threshold:.2f} | "
        f"Precision: {precision:.4f} | "
        f"Recall: {recall:.4f} | "
        f"F1: {f1:.4f}"
    )

    if f1 > best_f1:

        best_f1 = f1
        best_threshold = threshold


# Save selected threshold
joblib.dump(
    best_threshold,
    "models/optimal_threshold.pkl"
)


print("\nOptimal Threshold:", best_threshold)
print("Best Validation F1:", best_f1)


# ------------------------------------------------
# FINAL EVALUATION ON TEST DATA
# ------------------------------------------------

test_predictions = (
    test_probabilities >= best_threshold
).astype(int)


print("\nFINAL TEST RESULTS")

print(
    "Accuracy:",
    accuracy_score(y_test, test_predictions)
)

print(
    "Precision:",
    precision_score(
        y_test,
        test_predictions,
        zero_division=0
    )
)

print(
    "Recall:",
    recall_score(
        y_test,
        test_predictions,
        zero_division=0
    )
)

print(
    "F1 Score:",
    f1_score(
        y_test,
        test_predictions,
        zero_division=0
    )
)


print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        test_predictions
    )
)


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        test_predictions,
        zero_division=0
    )
)


print(
    "ROC-AUC:",
    roc_auc_score(
        y_test,
        test_probabilities
    )
)


print(
    "PR-AUC:",
    average_precision_score(
        y_test,
        test_probabilities
    )
)


print("\nEvaluation completed!")
