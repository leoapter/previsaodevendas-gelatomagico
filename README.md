# 🍦 Gelato Mágico - Previsão de Vendas com Machine Learning na Azure

Este projeto faz parte dos desafios e atividades do Bootcamp DIO - Microsoft Certification Challenge #3 DP-100, cujo objetivo é prever vendas de sorvete com base na temperatura do dia, utilizando **Machine Learning**, a plataforma **Microsoft Azure Machine Learning**, e boas práticas de **MLOps** como reprodutibilidade, versionamento com **MLflow**, e deploy em ambiente cloud.

---

## 📌 Cenário

Imagine que você é dono da sorveteria **Gelato Mágico**, localizada em uma cidade litorânea. Percebeu que as vendas de sorvetes estão diretamente relacionadas com a **temperatura** do dia. Porém, sem um planejamento, pode acabar:

- ❌ Produzindo mais sorvetes do que o necessário (desperdício);
- ❌ Produzindo menos do que a demanda (perda de oportunidade de venda).

---

### 🎯 Objetivo

Desenvolver um modelo de regressão linear para prever a quantidade de sorvetes vendidos com base na temperatura. O projeto envolve:

✅ Treinamento do modelo de Machine Learning  
✅ Registro e gerenciamento com **MLflow** na **Azure Machine Learning**  
✅ Deploy em ambiente **cloud**  
✅ Pipeline estruturado e reprodutível  

---

## 🗂️ Estrutura do Projeto
````
gelato_magico_ml/
│
├── inputs/
│   └── frases.txt
│
├── notebooks/
│   └── modelo_regressao.ipynb
│
├── src/
│   └── pipeline.py
│   └── previsao.py
│
├── models/
│   └── modelo_mlflow/
│
├── readme.md
└── requirements.txt
````


---

## 🧾 Arquivo de entrada `frases.txt` (em `inputs/`)
Essas sentenças foram usadas como inspiração para o contexto do modelo e compreensão de padrões.

```txt
Hoje está muito quente, acho que venderemos muitos sorvetes.
A temperatura caiu, talvez a demanda por sorvetes seja menor.
Quando o dia está ensolarado, a sorveteria bomba!
Com dias frios, o movimento tende a diminuir bastante.
Temperaturas acima de 30 graus geralmente indicam alta demanda.
````

---

## 🔁 Pipeline do Modelo
````
# src/pipeline.py
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd

# Dados de exemplo
df = pd.DataFrame({
    'temperatura': [20, 22, 25, 27, 30, 33, 35],
    'vendas': [120, 135, 150, 160, 185, 200, 210]
})

# Split
X = df[['temperatura']]
y = df['vendas']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Modelo
model = LinearRegression()
model.fit(X_train, y_train)
````

---

## ☁️ Registro com MLflow na Azure ML
Configuração do ambiente no portal da Azure Machine Learning.
````
import mlflow
import mlflow.sklearn
from azureml.core import Workspace

# Conectar com workspace da Azure
ws = Workspace.from_config()

mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())
mlflow.set_experiment("Previsao_Vendas_Sorvete")

with mlflow.start_run():
    model = LinearRegression().fit(X_train, y_train)
    mlflow.sklearn.log_model(model, "modelo_sorvete")
    mlflow.log_metric("score", model.score(X_test, y_test))
````

---

## 🔧 Requisitos (requirements.txt)
````
pandas
scikit-learn
mlflow
azureml-core
azureml-sdk
joblib
````

---

## 🚀 Deploy do Modelo na Azure
Com o modelo registrado, usar o portal da Azure para:

Criar um endpoint REST

Publicar o modelo como um serviço web (real-time inference)

Conectar à uma aplicação

---

## 🧠 Insights e Possibilidades
A temperatura realmente é um excelente preditor para vendas de sorvete.

O uso de Azure ML + MLflow facilita o versionamento e automação do deploy.

A pipeline pode ser estendida com:

📅 Variáveis sazonais (fim de semana, feriado)

☁️ Condições climáticas adicionais (chuva, umidade, vento)

📍 Dados geográficos (proximidade de praias, eventos locais)

---

✅ Conclusão
Este projeto mostra como é possível aplicar conceitos simples de regressão linear para resolver problemas do mundo real, combinando inteligência artificial com a infraestrutura de nuvem da Microsoft Azure. A abordagem garante reprodutibilidade, escalabilidade e inteligência de negócio para qualquer operação de vendas impactada pelo clima.

---
