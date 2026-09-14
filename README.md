# Detecção de transações fraudulentas com aprendizado de máquina

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

Relacionamos as tabelas de transação e identidade usando `TransactionID`. Após a junção, o conjunto ficou com 590.540 transações e 434 colunas: 403 numéricas e 31 categóricas. Foram encontradas 414 colunas com algum valor ausente.

Separamos `isFraud` como variável-alvo, com `0` para transação normal e `1` para transação fraudulenta, e removemos `TransactionID` das variáveis usadas pelos modelos. Os dados foram divididos em 80% para treino e 20% para validação, de forma estratificada e usando `random_state=42`. O treino ficou com 472.432 linhas e a validação com 118.108 linhas.

Os valores ausentes numéricos foram preenchidos pela mediana calculada no conjunto de treino. Os valores ausentes categóricos receberam o texto `ausente`. As colunas anonimizadas, como `V1` a `V339` e `id_01` a `id_38`, foram mantidas, pois não foram removidas apenas por terem nomes genéricos.

## Quais modelos foram utilizados

Testamos cinco modelos usando a mesma preparação, divisão de treino e validação e métrica ROC-AUC:

| modelo | preparação específica |
|---|---|
| `DummyClassifier` | linha de base com a estratégia `prior` |
| Regressão logística | normalização das variáveis numéricas e one-hot encoding das categóricas |
| Random Forest | codificação ordinal das categóricas |
| LightGBM | codificação ordinal das categóricas |
| CatBoost | tratamento direto das variáveis categóricas |

Nos modelos de classificação, exceto o `DummyClassifier`, foi usado balanceamento de classes, pois apenas 3,50% das transações eram fraudulentas.

## Resultados

A métrica principal usada será a **ROC-AUC** (*Receiver Operating Characteristic - Area Under the Curve*). Ela mede o quanto o modelo consegue separar transações normais de transações fraudulentas.

Essa métrica foi escolhida porque a base possui muito mais transações normais do que fraudulentas. Nesse caso, a acurácia pode dar uma impressão enganosa do desempenho. O ROC-AUC avalia a capacidade do modelo de ordenar as transações mais suspeitas, sem depender de um único ponto de corte.

O valor do ROC-AUC varia de 0 a 1: quanto mais próximo de 1, melhor a separação entre as classes. Um valor próximo de 0,5 indica um resultado semelhante ao acaso.

Os resultados abaixo foram obtidos na mesma divisão estratificada de treino e validação:

| modelo | ROC-AUC | tempo de treinamento e previsão |
|---|---:|---:|
| `DummyClassifier` | 0,5000 | 0,04 s |
| Regressão logística | 0,8617 | 258,11 s |
| Random Forest | 0,9127 | 23,67 s |
| LightGBM | 0,9137 | 15,84 s |
| CatBoost | 0,8915 | 56,77 s |

O `DummyClassifier` representa a linha de base: seu ROC-AUC de 0,5000 indica um resultado equivalente ao acaso. O melhor resultado foi do LightGBM, com ROC-AUC de 0,9137, ligeiramente acima do Random Forest e com menor tempo de execução. Por isso, ele foi escolhido como o modelo principal da primeira rodada.

## O que não funcionou

A primeira execução da regressão logística apresentou falta de memória durante a identificação dos tipos de coluna. Ajustamos a preparação para consultar os tipos sem copiar a tabela inteira. Depois disso, o modelo foi executado, mas atingiu o limite de 100 iterações e apresentou um aviso de não convergência. Além disso, foi o modelo mais lento da rodada.

O CatBoost executou normalmente, mas seu ROC-AUC de 0,8915 ficou abaixo dos resultados do Random Forest e do LightGBM. Por isso, não foi escolhido como modelo principal nesta primeira comparação.

## Limitação honesta do trabalho

A validação foi feita com uma divisão aleatória estratificada. Ela é simples e mantém a proporção de fraudes, mas não representa perfeitamente o uso do modelo para prever transações futuras. Além disso, o conjunto de teste do Kaggle não possui os rótulos públicos de fraude, então o resultado apresentado foi medido apenas no conjunto de validação.
