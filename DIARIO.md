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

