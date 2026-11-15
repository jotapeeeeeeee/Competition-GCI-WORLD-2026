# 🏦 GCI World 2026 - Home Credit Default Risk
### Juan Pablo Toledo (@jotapeeeeeeee)

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![LightGBM](https://img.shields.io/badge/LightGBM-green)
![XGBoost](https://img.shields.io/badge/XGBoost-red)
![CatBoost](https://img.shields.io/badge/CatBoost-yellow)
![Status](https://img.shields.io/badge/Status-En%20mejora%20continua-orange)

Este repositorio contiene mi solución para la competición **GCI World 2026 (Home Credit Default Risk)** organizada por el **Matsuo-Iwasawa Lab de la Universidad de Tokio (UTokyo)**. El objetivo es predecir la probabilidad de que un cliente incurra en impago (default) en un préstamo, evaluado mediante la métrica **ROC AUC**.

## 🎯 Resultados Principales

Mi modelo final logró un **AUC de 0.76513** en la validación (OOF) y en el Leaderboard público, situándose en una posición competitiva.

| Modelo | OOF AUC (5-Fold) | Notas |
| :--- | :--- | :--- |
| Logistic Regression | ~0.660 | Baseline lineal con escalado |
| Random Forest | ~0.700 | Bagging con balanceo de clases |
| CatBoost | ~0.750 | Boosting robusto |
| LightGBM | 0.75692 | Boosting rápido y eficiente |
| XGBoost | 0.76037 | Boosting con regularización |
| **Ensemble (Optimizado)** | **0.76513** | **Pesos optimizados vía Nelder-Mead** |

> **⚠️ Nota sobre el Leaderboard:** Durante la competición se observaron puntuaciones cercanas a `0.999`. En problemas reales de riesgo crediticio, un AUC perfecto es una señal de *Data Leakage* (fuga de datos) o manipulación de la métrica. Este proyecto se enfoca en construir un modelo **realista, robusto y generalizable**, evitando trampas y manteniendo la integridad analítica.

## 🛠️ Metodología (Pipeline End-to-End)

El código (`solution.py`) está diseñado para ser **100% reproducible**. Fijando la semilla (`SEED = 42`), los resultados se mantienen constantes entre ejecuciones. El proceso se divide en:

1. **Data Cleaning & Feature Engineering:**
   - Tratamiento de valores centinela (ej. `DAYS_EMPLOYED = 365243`).
   - Eliminación de outliers (ej. `OWN_CAR_AGE >= 60`).
   - Creación de ratios financieros (`CREDIT_INCOME_RATIO`, `ANNUITY_INCOME_RATIO`).
   - Agregaciones de variables de alta predictibilidad (`EXT_SOURCE_MEAN`, `EXT_SOURCE_PROD`).
   - Imputación de nulos con indicadores de missing.
2. **Encoding de Categóricas:**
   - Uso de categorías unificadas de *train* y *test* para evitar desalineaciones.
   - Tratamiento de valores `XNA` como `Unknown`.
3. **Validación Cruzada:**
   - Uso de `StratifiedKFold` (5 folds) para mantener la proporción de la clase minoritaria (8% de impagos).
4. **Modelado & Ensembling:**
   - Entrenamiento de 5 modelos distintos para asegurar la **diversidad** (vital en ensembles).
   - Uso de *Early Stopping* para evitar sobreajuste.
   - Extracción de predicciones *Out-Of-Fold* (OOF) para calcular los pesos del ensemble de forma honesta.
   - Optimización de pesos mediante `scipy.optimize.minimize` aplicando *Rank Normalization*.

## 📂 Estructura del Repositorio

```text
├── input/                  # Carpeta para colocar los datos (NO INCLUIDOS en el repo)
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
├── output/                 # Carpeta generada automáticamente
│   ├── submission.csv      # Predicciones finales
│   └── tutorial.ipynb      # Notebook auto-generado
├── solution.py             # Script principal end-to-end (genera el submission y el zip)
├── tutorial.zip            # Código comprimido para entrega en Omnicampus
└── README.md
