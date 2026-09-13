import time

import numpy as np
from pandas.api.types import is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from preparacao import carregar_dados, preparar_dados


def main():
    dados = carregar_dados()
    X_train, X_valid, y_train, y_valid = preparar_dados(dados)

    numeric_columns = X_train.columns[
        X_train.dtypes.map(is_numeric_dtype)
    ]
    categorical_columns = X_train.columns[~X_train.columns.isin(numeric_columns)]

    preprocessamento = ColumnTransformer(
        transformers=[
            (
                "categoricas",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    dtype=np.float32,
                ),
                categorical_columns,
            ),
        ],
        remainder="passthrough",
        sparse_threshold=0,
    )

    modelo = Pipeline(
        steps=[
            ("preprocessamento", preprocessamento),
            (
                "classificador",
                RandomForestClassifier(
                    n_estimators=50,
                    max_depth=20,
                    max_samples=0.5,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    inicio = time.perf_counter()
    modelo.fit(X_train, y_train)
    probabilidades = modelo.predict_proba(X_valid)[:, 1]
    roc_auc = roc_auc_score(y_valid, probabilidades)
    tempo = time.perf_counter() - inicio

    quantidade_codificada = len(
        modelo.named_steps["preprocessamento"].get_feature_names_out()
    )

    print("RANDOM FOREST")
    print("Codificação: ordinal para categóricas")
    print("Configuração: 50 árvores, profundidade máxima 20")
    print("Amostragem: 50% das linhas por árvore")
    print("Balanceamento: class_weight='balanced'")
    print(f"Colunas após a codificação: {quantidade_codificada:,}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"Tempo de treinamento e previsão: {tempo:.2f} segundos")
    print(f"Divisão usada: {len(X_train):,} treino / {len(X_valid):,} validação")


if __name__ == "__main__":
    main()
