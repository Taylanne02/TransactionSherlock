# Título

## Como rodar

O dataset não é enviado ao GitHub porque seus arquivos são muito grandes. Para executar o projeto, baixe os dados no [Kaggle](https://www.kaggle.com/competitions/ieee-fraud-detection/overview) e monte a pasta localmente.

1. Acesse o link do Kaggle, faça login e baixe os arquivos da competição **IEEE-CIS Fraud Detection**.
2. Extraia os arquivos baixados.
3. Dentro do projeto, crie a pasta `dados/IEEE-CIS/`.
4. Coloque dentro dela estes quatro arquivos:

```text
dados/
└── IEEE-CIS/
    ├── train_transaction.csv
    ├── train_identity.csv
    ├── test_transaction.csv
    └── test_identity.csv
```

A pasta `dados/IEEE-CIS/` está no `.gitignore` e não deve ser enviada ao GitHub. Depois dessa montagem, os arquivos podem ser usados pelo código do projeto.

## Problema e porque importa

Problema escolhido: `Detectar transações suspeitas em uma base pública de fraude`

Fraudes em transações podem causar prejuízo para empresas e clientes. Um modelo de aprendizado de máquina pode ajudar a identificar transações com maior chance de fraude para que sejam analisadas ou bloqueadas. O problema será tratado como uma classificação: cada transação será classificada como normal ou fraudulenta.

## De onde vieram os dados, e quantos são

Os dados são do dataset público IEEE-CIS Fraud Detection, disponibilizado no Kaggle. Eles foram organizados na pasta `dados/IEEE-CIS/` e ocupam aproximadamente 1,35 GB.

O dataset possui quatro arquivos:

- `train_transaction.csv`: 590.540 transações, incluindo a coluna `isFraud`, que informa se a transação é fraude.
- `train_identity.csv`: 144.233 registros de identidade relacionados às transações de treino.
- `test_transaction.csv`: 506.691 transações para teste.
- `test_identity.csv`: 141.907 registros de identidade relacionados às transações de teste.

As tabelas de transação e identidade possuem informações como valor da compra, produto, cartão, endereço e dispositivo. A identificação usada para relacionar os registros é `TransactionID`.

Link: [https://www.kaggle.com/competitions/ieee-fraud-detection/overview](https://www.kaggle.com/competitions/ieee-fraud-detection/overview)

## O que foi feito antes de modelar

Os dados foram separados em arquivos de treino e teste. Antes de treinar os modelos, a tabela de transações será relacionada à tabela de identidade usando `TransactionID`. A coluna `isFraud` será usada como variável-alvo, com `0` para transação normal e `1` para transação fraudulenta.

Também serão verificados os valores ausentes e as colunas categóricas. Esses dados precisarão ser preparados para que possam ser utilizados pelos modelos de aprendizado de máquina. Como o conjunto de teste não possui a coluna `isFraud`, os modelos serão comparados usando uma divisão de validação feita a partir dos dados de treino.

Não há colunas literalmente sem nome nos arquivos. Algumas possuem nomes genéricos ou anonimizados, como `V1` a `V339` e `id_01` a `id_38`. Elas não serão descartadas apenas por não terem nomes descritivos, pois ainda podem ajudar na identificação de fraudes. A remoção de colunas só será feita se a análise mostrar que elas estão totalmente vazias ou são inadequadas para a modelagem.

## Quais modelos foram utilizados

## Resultados

A métrica principal usada será a **ROC-AUC** (*Receiver Operating Characteristic - Area Under the Curve*). Ela mede o quanto o modelo consegue separar transações normais de transações fraudulentas.

Essa métrica foi escolhida porque a base possui muito mais transações normais do que fraudulentas. Nesse caso, a acurácia pode dar uma impressão enganosa do desempenho. O ROC-AUC avalia a capacidade do modelo de ordenar as transações mais suspeitas, sem depender de um único ponto de corte.

O valor do ROC-AUC varia de 0 a 1: quanto mais próximo de 1, melhor a separação entre as classes. Um valor próximo de 0,5 indica um resultado semelhante ao acaso.

## O que não funcionou

## Limitação honesta do trabalho
