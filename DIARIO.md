## 31/08

Escolha do tema, com o dataset IEEE-CIS Fraud Detection e criação do repositório.

## 12/09 22:48

Revisamos o objetivo do trabalho e confirmamos o uso do dataset IEEE-CIS Fraud Detection, disponível no Kaggle. Os dados estão organizados em quatro arquivos na pasta `dados/IEEE-CIS/`: transações e identidades para treino e teste.

Decidimos tratar o problema como uma classificação binária, usando `isFraud` como variável-alvo. A métrica escolhida foi ROC-AUC, pois a base possui muito mais transações normais do que fraudulentas e a acurácia poderia esconder um resultado ruim para fraudes.

Verificamos que não existem colunas literalmente sem nome. As colunas com nomes genéricos, como `V1` a `V339` e `id_01` a `id_38`, são anonimizadas e não serão descartadas apenas por causa do nome.

Também atualizamos o `README.md` com a origem e a quantidade dos dados, a preparação prevista e a explicação do ROC-AUC.

Próximo passo: verificar os valores ausentes e as colunas categóricas, relacionar as tabelas usando `TransactionID` e preparar uma divisão de validação a partir dos dados de treino.

