## 31/08

Escolha do tema, com o dataset IEEE-CIS Fraud Detection e criação do repositório.

## 12/09 22:48

Revisamos o objetivo do trabalho e confirmamos o uso do dataset IEEE-CIS Fraud Detection, disponível no Kaggle. Os dados estão organizados em quatro arquivos na pasta `dados/IEEE-CIS/`: transações e identidades para treino e teste.

Decidimos tratar o problema como uma classificação binária, usando `isFraud` como variável-alvo. A métrica escolhida foi ROC-AUC, pois a base possui muito mais transações normais do que fraudulentas e a acurácia poderia esconder um resultado ruim para fraudes.

Verificamos que não existem colunas literalmente sem nome. As colunas com nomes genéricos, como `V1` a `V339` e `id_01` a `id_38`, são anonimizadas e não serão descartadas apenas por causa do nome.

Também atualizamos o `README.md` com a origem e a quantidade dos dados, a preparação prevista e a explicação do ROC-AUC.

Próximo passo: verificar os valores ausentes e as colunas categóricas, relacionar as tabelas usando `TransactionID` e preparar uma divisão de validação a partir dos dados de treino.

## 13/09 17:50

Criamos `scripts/01_diagnostico.py` e executamos o diagnóstico com os dados reais de treino. A tabela de transações possui 590.540 linhas e 394 colunas. A tabela de identidade possui 144.233 linhas e 41 colunas. Após a junção por `TransactionID`, foram mantidas 590.540 linhas e o conjunto passou a ter 434 colunas.

Encontramos 569.877 transações normais e 20.663 fraudulentas, o que corresponde a 3,50% de fraudes. O conjunto possui 403 colunas numéricas e 31 categóricas. Das 434 colunas, 414 possuem algum valor ausente. As maiores proporções de ausência estão em `id_24`, `id_25`, `id_07`, `id_08`, `id_21`, `id_26`, `id_27`, `id_23`, `id_22` e `dist2`.

Próximo passo: separar `isFraud` como alvo, remover `TransactionID`, dividir os dados em treino e validação e definir o preenchimento simples dos valores ausentes.

## 13/09 18:17

Criamos `scripts/preparacao.py` para preparar a base comum dos modelos. `isFraud` foi separado como alvo e `TransactionID` foi removido das variáveis. Os dados foram divididos de forma estratificada em 80% para treino e 20% para validação, usando `random_state=42`.

Na execução com os dados reais, o treino ficou com 472.432 linhas e 16.530 fraudes. A validação ficou com 118.108 linhas e 4.133 fraudes. As duas partes mantiveram a proporção de 3,50% de fraudes e ficaram sem valores ausentes: valores numéricos foram preenchidos pela mediana do treino e valores categóricos receberam `ausente`.

Próximo passo: criar as preparações específicas e treinar o `DummyClassifier`, a regressão logística, a floresta aleatória, o LightGBM e o CatBoost usando a mesma divisão.

## 13/09 18:52

Treinamos o `DummyClassifier` usando a estratégia `prior`, a mesma divisão de treino e validação e a métrica ROC-AUC. O resultado foi ROC-AUC = 0,5000, com 0,04 segundo de execução.

Esse resultado foi usado como linha de base: o modelo atribui a mesma probabilidade de fraude às transações e, por isso, não consegue separá-las melhor que o acaso.

Próximo passo: treinar a regressão logística na mesma divisão e comparar o resultado com esta linha de base.

## 13/09 19:04

Treinamos a `LogisticRegression` usando a mesma preparação, divisão e métrica do `DummyClassifier`. As colunas categóricas foram transformadas com one-hot encoding, as numéricas foram normalizadas e foi usado `class_weight="balanced"` por causa do desbalanceamento.

A primeira execução falhou por falta de memória durante a identificação dos tipos de coluna. Ajustamos a preparação para consultar os tipos sem copiar a tabela inteira e executamos novamente.

O modelo gerou ROC-AUC = 0,8617 em 258,11 segundos, com 2.733 colunas após a codificação. A otimização atingiu o limite de 100 iterações e apresentou um aviso de não convergência; por isso, este resultado será tratado como a primeira tentativa da regressão logística, não como uma configuração final.

Próximo passo: treinar o `RandomForest` usando a mesma divisão e comparar com o `DummyClassifier` e a regressão logística.

## 13/09 19:13

Treinamos o `RandomForest` usando a mesma preparação, divisão e métrica dos modelos anteriores. As colunas categóricas receberam codificação ordinal. Usamos 50 árvores, profundidade máxima 20, 50% das linhas por árvore e `class_weight="balanced"`.

O modelo gerou ROC-AUC = 0,9127 em 23,67 segundos. O resultado foi melhor que o `DummyClassifier` (0,5000) e a `LogisticRegression` (0,8617). Nesta primeira tentativa, a floresta foi mais rápida que a regressão logística e apresentou o melhor resultado até agora.

Próximo passo: treinar o `LightGBM` usando a mesma divisão e comparar os resultados.

## 13/09 21:53

Treinamos o `LightGBM` usando a mesma preparação, divisão e métrica dos modelos anteriores. As colunas categóricas receberam codificação ordinal e foi usado `class_weight="balanced"`. A configuração inicial teve 100 árvores, `learning_rate=0,05` e `num_leaves=31`.

O modelo gerou ROC-AUC = 0,9137 em 15,84 segundos, com 432 colunas após a codificação. O resultado foi ligeiramente melhor que o `RandomForest` (0,9127), além de ter levado menos tempo, e superou a `LogisticRegression` (0,8617) e o `DummyClassifier` (0,5000).

Próximo passo: treinar o `CatBoost` usando a mesma divisão e comparar os resultados.

## 14/09 00:14

Treinamos o `CatBoost` usando a mesma preparação, divisão e métrica dos outros modelos. As 31 colunas categóricas foram informadas diretamente ao modelo, sem codificação ordinal ou one-hot. Usamos 100 iterações, `learning_rate=0,05`, profundidade 6 e `auto_class_weights="Balanced"`.

O modelo gerou ROC-AUC = 0,8915 em 56,77 segundos. Nesta configuração inicial, ficou abaixo do `LightGBM` e do `RandomForest`, mas acima da `LogisticRegression` e do `DummyClassifier`.

Resumo da primeira rodada:

| modelo | ROC-AUC | tempo |
|---|---:|---:|
| DummyClassifier | 0,5000 | 0,04 s |
| LogisticRegression | 0,8617 | 258,11 s |
| RandomForest | 0,9127 | 23,67 s |
| LightGBM | 0,9137 | 15,84 s |
| CatBoost | 0,8915 | 56,77 s |

Próximo passo: revisar os resultados da rodada e atualizar o `README.md` com os modelos, as métricas e a comparação com a linha de base.
