import time

from catboost import CatBoostClassifier
from pandas.api.types import is_numeric_dtype
from sklearn.metrics import roc_auc_score

from preparacao import carregar_dados, preparar_dados


def main():
    dados = carregar_dados()
    X_train, X_valid, y_train, y_valid = preparar_dados(dados)

    numeric_columns = X_train.columns[
        X_train.dtypes.map(is_numeric_dtype)
    ]
    categorical_columns = X_train.columns[~X_train.columns.isin(numeric_columns)]

    modelo = CatBoostClassifier(
        iterations=100,
        learning_rate=0.05,
        depth=6,
        auto_class_weights="Balanced",
        random_seed=42,
        eval_metric="AUC",
        verbose=False,
        allow_writing_files=False,
        thread_count=-1,
    )

    inicio = time.perf_counter()
    modelo.fit(X_train, y_train, cat_features=list(categorical_columns))
    probabilidades = modelo.predict_proba(X_valid)[:, 1]
    roc_auc = roc_auc_score(y_valid, probabilidades)
    tempo = time.perf_counter() - inicio

    print("CATBOOST")
    print("Codificação: categóricas tratadas diretamente pelo CatBoost")
    print("Configuração: 100 iterações, learning_rate=0.05, depth=6")
    print("Balanceamento: auto_class_weights='Balanced'")
    print(f"Colunas categóricas: {len(categorical_columns):,}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"Tempo de treinamento e previsão: {tempo:.2f} segundos")
    print(f"Divisão usada: {len(X_train):,} treino / {len(X_valid):,} validação")


if __name__ == "__main__":
    main()
