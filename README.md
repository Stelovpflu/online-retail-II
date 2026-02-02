📦 Online Retail II – Customer Segmentation & Abandono (Churn) Prediction System

---
🔍 Descripción del Proyecto

Proyecto end-to-end de Machine Learning orientado a negocio para segmentación de clientes y predicción de abandono, construido sobre datos transaccionales reales de retail.

La solución está diseñada con un enfoque business-first, priorizando la detección temprana de clientes en riesgo de abandono para habilitar estrategias de retención accionables y medibles.

Incluye:

Feature engineering a nivel cliente

Segmentación estratégica con KMeans

Modelo predictivo optimizado por recall

Artefactos listos para despliegue y consumo vía app o API

---
🚀 Simulador 👉 Streamlit App

(enlace a agregar)

Funcionalidades:

Ingreso manual de métricas del cliente

Identificación automática del segmento

Predicción de probabilidad de abandono

Recomendación de acción basada en un threshold alineado al negocio

---
🎯 Objetivo de Negocio

Detectar clientes con alta probabilidad de abandono

Priorizar esfuerzos de retención sobre clientes de mayor valor

Reducir pérdidas futuras asociadas a inacción tardía

Apoyar decisiones comerciales con modelos explicables

---
🧠 Preparación de Datos

Dataset: Online Retail II

Transacciones inválidas filtradas (precio y cantidad positivos)

Agregación a nivel cliente

Enfoque de snapshot temporal para evitar data leakage

Definición de Abandono

Un cliente se considera en abandono si no ha realizado compras en los últimos 90 días.

---
🧩 Feature Engineering (Customer-Level)

Variables construidas:

recency_days

frequency

monetary

avg_order_value

tenure_days

purchase_velocity

Estas métricas capturan valor, frecuencia, temporalidad y dinámica de compra.

---
🧠 Segmentación de Clientes

Algoritmo: KMeans

Clusters: 4

Escalado: StandardScaler

Uso: análisis estratégico (no como input del modelo predictivo)

Segmentos generados:

High Value At Risk

Low Value Occasional

Mid Value Drifting

New / Unqualified

---
🤖 Modelo Predictivo

Algoritmo: Gradient Boosting Classifier

Ajuste de hiperparámetros mediante validación cruzada

Optimizado para maximizar recall en clientes en abandono

Entrenamiento final utilizando toda la información disponible

---
📊 Métricas Finales (Test)

ROC AUC: ≈ 0.82

Recall (abandono): ≈ 96%

Precision (abandono): ≈ 0.66

Trade-off alineado con un enfoque preventivo de retención.

---
🎯 Decision Threshold

Threshold seleccionado: 0.30

Optimizado para:

Minimizar falsos negativos

Detectar abandono temprano

Alineado con escenarios donde el costo de contacto es bajo frente al valor del cliente

---
📦 Artefactos Generados

gb_churn_pipeline.pkl – Modelo predictivo

kmeans_customer_segmentation.pkl – Segmentación de clientes

kmeans_scaler.pkl – Escalador del clustering

model_metadata.pkl – Metadata técnica y de negocio

Listo para:

API

Batch scoring

Dashboard ejecutivo

▶️ Ejecución Local
pip install -r requirements.txt
streamlit run app.py

---
⚠️ Notas

Este repositorio contiene el código de inferencia y despliegue.
El entrenamiento, validación y tuning del modelo forman parte del flujo de desarrollo offline.

---
👤 Autor

Steve Loveday
Data Scientist – Business Analytics & Predictive Modeling