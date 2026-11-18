
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split

from src.preprocessing import load_data, preprocess_data, scale_data
from src.model import CreditRiskNN

import os
import joblib


# Load dataset
df = load_data("data/Loan_default.csv")

X, y = preprocess_data(df)


# Split dataset: 70% training, 30% temporary
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)


# Split temporary data: 15% validation, 15% testing
X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)


print("Training samples:", len(X_train))
print("Validation samples:", len(X_val))
print("Testing samples:", len(X_test))


# Scale features using training data only
X_train, X_val, scaler = scale_data(
    X_train,
    X_val
)

X_test = scaler.transform(X_test)


# Convert training data to PyTorch tensors
X_train_tensor = torch.tensor(
    X_train,
    dtype=torch.float32
)

y_train_tensor = torch.tensor(
    y_train.values,
    dtype=torch.float32
).view(-1, 1)


# Initialize neural network
input_size = X_train.shape[1]

model = CreditRiskNN(input_size)


# Calculate positive class weight
positive_weight = torch.tensor(
    [(y_train == 0).sum() / (y_train == 1).sum()],
    dtype=torch.float32
)


# Loss function and optimizer
criterion = nn.BCEWithLogitsLoss(
    pos_weight=positive_weight
)

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)


# Train model
epochs = 500

for epoch in range(epochs):

    model.train()

    optimizer.zero_grad()

    outputs = model(X_train_tensor)

    loss = criterion(
        outputs,
        y_train_tensor
    )

    loss.backward()

    optimizer.step()

    if (epoch + 1) % 10 == 0:

        print(
            f"Epoch [{epoch+1}/{epochs}], "
            f"Loss: {loss.item():.4f}"
        )


# Save model and preprocessing artifacts
os.makedirs("models", exist_ok=True)

torch.save(
    model.state_dict(),
    "models/credit_risk_model.pth"
)

joblib.dump(
    scaler,
    "models/scaler.pkl"
)

# Save feature names and optimal threshold will be
# determined later using validation data.

joblib.dump(
    list(X.columns),
    "models/feature_columns.pkl"
)


print("\nModel training completed!")
print("Model and scaler saved successfully.")
