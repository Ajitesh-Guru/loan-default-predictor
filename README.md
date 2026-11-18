# Loan Default Prediction

A machine learning project that predicts the likelihood of loan default using borrower and loan-related information. The project implements a feed-forward neural network in PyTorch and addresses class imbalance using weighted binary cross-entropy loss.

## Project Overview

Credit risk assessment is an important part of lending operations. Financial institutions need to identify borrowers who are likely to default while minimizing the risk of incorrectly rejecting creditworthy applicants.

This project develops a binary classification model to predict whether a borrower will default on a loan (`1`) or repay it (`0`).

**Key highlights:**
- Processed a lending dataset containing approximately 255K records.
- Implemented a feed-forward neural network using PyTorch.
- Addressed class imbalance using class-weighted loss.
- Used stratified train-validation-test splitting.
- Optimized the classification threshold using validation F1-score.
- Evaluated the final model on an independent test dataset.

## Model Architecture

The project uses a feed-forward neural network with the following architecture:

| Layer | Configuration |
|---|---|
| Input layer | 10 features |
| Hidden layer 1 | 64 neurons, ReLU |
| Hidden layer 2 | 32 neurons, ReLU |
| Output layer | 1 neuron (logit) |

**Training configuration:**
- Framework: PyTorch
- Optimizer: Adam
- Learning rate: 0.001
- Loss function: `BCEWithLogitsLoss`
- Epochs: 500
- Class imbalance handling: Positive class weighting

The model outputs a raw logit. Sigmoid is applied during evaluation and inference to obtain the probability of loan default.

## Data Preprocessing

The preprocessing pipeline consists of the following steps:

1. **Feature selection:** Removed 7 columns that were excluded from model training.
2. **Categorical encoding:**
   - `HasMortgage`: Yes = 1, No = 0
   - `HasDependents`: Yes = 1, No = 0
   - `LoanPurpose`: Business = 0, Home = 1, Education = 2, Other = 3, Auto = 4
3. **Feature scaling:** Standardized 7 numerical features using `StandardScaler`.
4. **Target variable:** `Default`, representing whether the borrower defaulted.
5. **Dataset splitting:** Stratified 70% training, 15% validation, and 15% testing.

The scaler is fitted only on the training data and reused for validation, testing, and new predictions to prevent data leakage.

### Model Input Features

The neural network uses 10 input features:

- Age
- Income
- LoanAmount
- CreditScore
- InterestRate
- LoanTerm
- DTIRatio
- HasMortgage
- HasDependents
- LoanPurpose

## Handling Class Imbalance

Loan default is a minority class in the dataset. A model trained without addressing this imbalance may achieve high accuracy while failing to identify actual defaulters.

To address this, the project uses `BCEWithLogitsLoss` with a positive class weight calculated as:

`pos_weight = Number of Non-Defaulters / Number of Defaulters`

This increases the penalty for incorrectly classifying defaulters, encouraging the model to identify more positive cases.

## Threshold Optimization

Instead of using the default classification threshold of 0.5, the project searches thresholds from 0.20 to 0.80 using the validation dataset.

The threshold that maximizes the validation F1-score is selected and saved for inference.

**Selected threshold: 0.63**

The test dataset is kept separate from this optimization process and is used only for final performance evaluation.

## Model Performance

The final model was evaluated on the held-out test dataset using a threshold of 0.63.

| Metric | Test Performance |
|---|---:|
| Accuracy | 80.43% |
| Precision | 28.26% |
| Recall | 44.54% |
| F1-score | 34.58% |
| ROC-AUC | 0.7382 |
| PR-AUC | 0.3014 |

### Confusion Matrix

| | Predicted: Non-Default | Predicted: Default |
|---|---:|---:|
| Actual: Non-Default | 28,827 | 5,028 |
| Actual: Default | 2,467 | 1,981 |

### Key Observations

- The model identifies approximately **44.5% of actual loan defaulters**.
- Class weighting improves the model's ability to identify the minority default class.
- Threshold optimization provides a better precision-recall balance than relying on the default 0.5 threshold.
- ROC-AUC of 0.7382 indicates a reasonable ability to distinguish between default and non-default borrowers.

Accuracy alone is not sufficient for evaluating this problem because of class imbalance. In a real lending environment, the classification threshold should also account for the financial consequences of false approvals and false rejections.

## Project Structure

```text
credit-risk-neural-network/
│
├── data/
│   └── Loan_default.csv
│
├── models/
│   ├── credit_risk_model.pth
│   ├── scaler.pkl
│   ├── feature_columns.pkl
│   └── optimal_threshold.pkl
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd credit-risk-neural-network
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it:

**Windows**
```bash
venv\Scripts\activate
```

**Linux / macOS**
```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add the dataset

Place `Loan_default.csv` inside the `data/` directory.

### 5. Train the model

```bash
python -m src.train
```

### 6. Evaluate the model

```bash
python -m src.evaluate
```

### 7. Run predictions

```bash
python -m src.predict
```

The training and evaluation scripts generate and use the model artifacts stored in the `models/` directory. Ensure the scripts are configured to use the same dataset path and preprocessing pipeline.

## Technologies Used

- **Programming Language:** Python
- **Deep Learning:** PyTorch
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **Model Persistence:** Joblib

## Future Improvements

- Compare neural network performance against Logistic Regression and tree-based models.
- Experiment with hyperparameter tuning and alternative network architectures.
- Explore techniques such as SMOTE and undersampling for class imbalance.
- Evaluate business-specific classification thresholds using the financial costs of false positives and false negatives.
- Investigate model interpretability using SHAP or feature importance techniques.

## Conclusion

This project demonstrates an end-to-end credit risk classification pipeline, covering data preprocessing, neural network development, class imbalance handling, threshold optimization, and model evaluation. It highlights the importance of evaluating classification models using metrics beyond accuracy, particularly in financial applications where incorrectly classifying a borrower can have significant business consequences.