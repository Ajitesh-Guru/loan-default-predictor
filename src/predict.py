import torch
import joblib
import pandas as pd

from src.model import CreditRiskNN
from src.preprocessing import NUMERICAL_COLUMNS


scaler = joblib.load("models/scaler.pkl")

FEATURE_COLUMNS = [
    'Age', 'Income', 'LoanAmount', 'CreditScore',
    'InterestRate', 'LoanTerm', 'DTIRatio',
    'HasMortgage', 'HasDependents', 'LoanPurpose'
]

model = CreditRiskNN(input_size=len(FEATURE_COLUMNS))

model.load_state_dict(
    torch.load(
        "models/credit_risk_model.pth",
        map_location="cpu",
        weights_only=True
    )
)

model.eval()


def predict_default(input_data):

    df = pd.DataFrame([input_data])
    df['HasMortgage'] = df['HasMortgage'].map({
        'Yes': 1,
        'No': 0
    })

    df['HasDependents'] = df['HasDependents'].map({
        'Yes': 1,
        'No': 0
    })

    df['LoanPurpose'] = df['LoanPurpose'].map({
        'Business': 0,
        'Home': 1,
        'Education': 2,
        'Other': 3,
        'Auto': 4
    })

    feature_columns = [
        'Age', 'Income', 'LoanAmount', 'CreditScore',
        'InterestRate', 'LoanTerm', 'DTIRatio',
        'HasMortgage', 'HasDependents', 'LoanPurpose'
    ]

    df = df[feature_columns]
    X = scaler.transform(df)

    X_tensor = torch.tensor(
        X,
        dtype=torch.float32
    )

    with torch.no_grad():

        probability = model(X_tensor).item()

    prediction = 1 if probability > 0.5 else 0

    return {
        "prediction": prediction,
        "default_probability": probability
    }


if __name__ == "__main__":

    sample = {
        'Age': 35,
        'Income': 60000,
        'LoanAmount': 15000,
        'CreditScore': 700,
        'InterestRate': 8.5,
        'LoanTerm': 36,
        'DTIRatio': 0.3,
        'HasMortgage': 'Yes',
        'HasDependents': 'No',
        'LoanPurpose': 'Business'
    }

    result = predict_default(sample)

    print(result)