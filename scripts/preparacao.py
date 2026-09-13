from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "dados" / "IEEE-CIS"


def carregar_dados():
    transactions = pd.read_csv(DATA_DIR / "train_transaction.csv")
    identity = pd.read_csv(DATA_DIR / "train_identity.csv")
    return transactions.merge(identity, on="TransactionID", how="left")


def preparar_dados(dados, test_size=0.2, random_state=42):
    X = dados.drop(columns=["isFraud", "TransactionID"])
    y = dados["isFraud"]

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    numeric_columns = X_train.columns[
        X_train.dtypes.map(pd.api.types.is_numeric_dtype)
    ]
    categorical_columns = X_train.columns[~X_train.columns.isin(numeric_columns)]

    medians = X_train[numeric_columns].median()
    X_train.loc[:, numeric_columns] = (
        X_train[numeric_columns].fillna(medians).fillna(0)
    )
    X_valid.loc[:, numeric_columns] = (
        X_valid[numeric_columns].fillna(medians).fillna(0)
    )

    for column in categorical_columns:
        X_train.loc[:, column] = X_train[column].astype("object").where(
            X_train[column].notna(), "ausente"
        )
        X_valid.loc[:, column] = X_valid[column].astype("object").where(
            X_valid[column].notna(), "ausente"
        )

    return X_train, X_valid, y_train, y_valid


def main():
    dados = carregar_dados()
    X_train, X_valid, y_train, y_valid = preparar_dados(dados)

    print("PREPARAÇÃO DOS DADOS")
    print(f"Treino: {X_train.shape[0]:,} linhas x {X_train.shape[1]} colunas")
    print(f"Validação: {X_valid.shape[0]:,} linhas x {X_valid.shape[1]} colunas")
    print(f"Fraudes no treino: {y_train.sum():,} ({y_train.mean() * 100:.2f}%)")
    print(f"Fraudes na validação: {y_valid.sum():,} ({y_valid.mean() * 100:.2f}%)")
    print(f"Valores ausentes no treino: {X_train.isna().sum().sum():,}")
    print(f"Valores ausentes na validação: {X_valid.isna().sum().sum():,}")
    print("Divisão: 80% treino e 20% validação, estratificada por isFraud")
    print("Tratamento: mediana para numéricas e 'ausente' para categóricas")


if __name__ == "__main__":
    main()
