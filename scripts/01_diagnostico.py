from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "dados" / "IEEE-CIS"


def main():
    transaction_path = DATA_DIR / "train_transaction.csv"
    identity_path = DATA_DIR / "train_identity.csv"

    transactions = pd.read_csv(transaction_path)
    identity = pd.read_csv(identity_path)

    merged = transactions.merge(identity, on="TransactionID", how="left")

    fraud_counts = transactions["isFraud"].value_counts().sort_index()
    fraud_rate = transactions["isFraud"].mean() * 100
    categorical_columns = merged.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()
    missing = merged.isna().mean().mul(100).sort_values(ascending=False)
    missing = missing[missing > 0]

    print("DIAGNÓSTICO DOS DADOS DE TREINO")
    print(f"Transações: {transactions.shape[0]:,} linhas x {transactions.shape[1]} colunas")
    print(f"Identidade: {identity.shape[0]:,} linhas x {identity.shape[1]} colunas")
    print(f"Após a junção: {merged.shape[0]:,} linhas x {merged.shape[1]} colunas")
    print()

    print("DISTRIBUIÇÃO DO ALVO")
    print(f"Transações normais (0): {fraud_counts.get(0, 0):,}")
    print(f"Transações fraudulentas (1): {fraud_counts.get(1, 0):,}")
    print(f"Proporção de fraudes: {fraud_rate:.2f}%")
    print()

    print("TIPOS DE COLUNA")
    print(f"Colunas numéricas: {merged.select_dtypes(include=['number']).shape[1]}")
    print(f"Colunas categóricas: {len(categorical_columns)}")
    print(f"Exemplos de categóricas: {', '.join(categorical_columns[:10])}")
    print()

    print("VALORES AUSENTES")
    print(f"Colunas com algum valor ausente: {len(missing)}")
    print("Maiores proporções de ausência:")
    for column, percentage in missing.head(10).items():
        print(f"- {column}: {percentage:.2f}%")


if __name__ == "__main__":
    main()
