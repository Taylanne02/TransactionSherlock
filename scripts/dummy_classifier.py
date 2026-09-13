import time

from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_auc_score

from preparacao import carregar_dados, preparar_dados


def main():
    dados = carregar_dados()
    X_train, X_valid, y_train, y_valid = preparar_dados(dados)

    modelo = DummyClassifier(strategy="prior")

    inicio = time.perf_counter()
    modelo.fit(X_train, y_train)
    probabilidades = modelo.predict_proba(X_valid)[:, 1]
    roc_auc = roc_auc_score(y_valid, probabilidades)
    tempo = time.perf_counter() - inicio

    print("DUMMY CLASSIFIER")
    print("Estratégia: prior")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"Tempo de treinamento e previsão: {tempo:.2f} segundos")
    print(f"Divisão usada: {len(X_train):,} treino / {len(X_valid):,} validação")


if __name__ == "__main__":
    main()
