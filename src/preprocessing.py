import pandas as pd
from sklearn.preprocessing import StandardScaler

DROP_COLUMNS = [
    'LoanID',
    'MonthsEmployed',
    'NumCreditLines',
    'Education',
    'EmploymentType',
    'MaritalStatus',
    'HasCoSigner'
]

NUMERICAL_COLUMNS = [
    'Age',
    'Income',
    'LoanAmount',
    'CreditScore',
    'InterestRate',
    'LoanTerm',
    'DTIRatio'
]

CATEGORICAL_COLUMNS = [
    'HasMortgage',
    'HasDependents',
    'LoanPurpose'
]

TARGET_COLUMN = 'Default'


def load_data(filepath):
    df = pd.read_csv(filepath)
    return df


def preprocess_data(df):
    df = df.copy()

    df.drop(columns=DROP_COLUMNS, inplace=True)

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

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    return X, y


def scale_data(X_train, X_test):
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, scaler