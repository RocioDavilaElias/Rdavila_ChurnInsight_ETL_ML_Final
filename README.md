# 🎯 ChurnInsight - Predicción de Cancelación de Clientes Netflix

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/19m1OyDlmwmqMZ4BplVcG4vqvnuR56UlB?usp=sharing)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📋 Tabla de Contenidos

1. [Descripción del Proyecto](#-descripción-del-proyecto)
2. [Contexto del Problema](#-contexto-del-problema)
3. [Dataset Utilizado](#-dataset-utilizado)
4. [Pipeline ETL - Paso a Paso](#-pipeline-etl---paso-a-paso)
5. [Análisis Exploratorio de Datos (EDA)](#-análisis-exploratorio-de-datos-eda)
6. [Detección de Data Leakage](#-detección-de-data-leakage---problema-crítico)
7. [Selección de Variables](#-selección-de-variables)
8. [Preprocesamiento de Datos](#-preprocesamiento-de-datos)
9. [Entrenamiento de Modelos](#-entrenamiento-de-modelos)
10. [Evaluación y Resultados](#-evaluación-y-resultados)
11. [Análisis de Impacto Financiero](#-análisis-de-impacto-financiero)
12. [Cómo Usar el Modelo](#-cómo-usar-el-modelo)
13. [Integración con API](#-integración-con-api)
14. [Conclusiones](#-conclusiones)

---

## 📖 Descripción del Proyecto

**ChurnInsight** es un modelo de Machine Learning diseñado para predecir la probabilidad de que un cliente de Netflix cancele su suscripción (churn). 

### ¿Qué es Churn?

El **churn** o "tasa de cancelación" es cuando un cliente deja de usar un servicio. En el caso de Netflix, sería un usuario que cancela su suscripción. Predecir qué clientes van a cancelar nos permite contactarlos antes y ofrecerles incentivos para que se queden.

### Objetivo Principal

Desarrollar un modelo capaz de:
- ✅ Identificar clientes en riesgo de cancelación
- ✅ Permitir intervenciones proactivas de retención
- ✅ Maximizar el Lifetime Value (LTV) de la base de usuarios
- ✅ Generar valor económico real para el negocio

**🔗 Notebook completo:** [Abrir en Google Colab](https://colab.research.google.com/drive/19m1OyDlmwmqMZ4BplVcG4vqvnuR56UlB?usp=sharing)

---

## 🏢 Contexto del Problema

La industria del entretenimiento por suscripción enfrenta un desafío constante: **reducir la pérdida de clientes** en un entorno altamente competitivo.

### ¿Por qué es importante?

| Dato | Impacto |
|------|---------|
| Costo de adquirir cliente nuevo | 5-25 veces más caro que retener uno existente |
| Tasa de churn típica en streaming | 2-5% mensual |
| Impacto en ingresos | Cada cliente perdido = pérdida de ingresos futuros |

### Beneficio de predecir churn

Si podemos identificar clientes que van a cancelar **antes** de que lo hagan, podemos:
1. Contactarlos proactivamente
2. Ofrecerles descuentos o beneficios
3. Resolver sus problemas
4. Retenerlos y mantener sus pagos futuros

---

## 📊 Dataset Utilizado

Se utiliza el dataset **"Netflix Customer Churn"** disponible en Kaggle.

### Características Generales

| Característica | Valor |
|----------------|-------|
| Total de registros | 5,000 clientes |
| Total de columnas | 14 variables |
| Variable objetivo | `churned` (1 = canceló, 0 = permaneció) |
| Tasa de churn | 50.3% |
| Valores nulos | 0 (dataset limpio) |

### Variables Originales del Dataset

| Variable | Descripción | Tipo |
|----------|-------------|------|
| `customer_id` | Identificador único del cliente | String |
| `age` | Edad del cliente (18-70 años) | Numérica |
| `gender` | Género (Male, Female, Other) | Categórica |
| `subscription_type` | Tipo de plan (Basic, Standard, Premium) | Categórica |
| `watch_hours` | Horas totales de visualización | Numérica |
| `last_login_days` | Días desde el último login | Numérica |
| `region` | Región geográfica (6 regiones) | Categórica |
| `device` | Dispositivo principal (5 tipos) | Categórica |
| `monthly_fee` | Cuota mensual ($8.99-$17.99) | Numérica |
| `payment_method` | Método de pago (5 métodos) | Categórica |
| `number_of_profiles` | Número de perfiles en la cuenta | Numérica |
| `avg_watch_time_per_day` | Promedio de horas diarias | Numérica |
| `favorite_genre` | Género favorito (7 géneros) | Categórica |
| `churned` | Si el cliente canceló (1) o no (0) | Booleana |

---

## 🔄 Pipeline ETL - Paso a Paso

ETL significa **Extract, Transform, Load** (Extraer, Transformar, Cargar). Es el proceso de preparar los datos para el modelo.

### Paso 1: Extracción (Extract)

```python
# Cargamos el dataset desde GitHub
url = "https://raw.githubusercontent.com/.../netflix_churn.csv"
df = pd.read_csv(url)
```

**¿Qué hicimos?**
- Cargamos el dataset directamente desde un repositorio GitHub
- Esto garantiza que cualquier persona pueda reproducir el análisis
- Verificamos que se cargaron 5,000 registros correctamente

### Paso 2: Transformación (Transform)

**2.1 Creación de identificador público:**
```python
# Crear public_id anonimizado con hash SHA-256
df['public_id'] = df['customer_id'].apply(
    lambda x: "CUS-" + hashlib.sha256(x.encode()).hexdigest()[:8].upper()
)
```

**¿Por qué?** Para proteger la identidad de los clientes pero mantener trazabilidad.

**2.2 Conversión de tipos de datos:**
```python
# Convertir categóricas a tipo 'category' (ahorra memoria)
for col in ['gender', 'subscription_type', 'region', 'device', 'payment_method']:
    df[col] = df[col].astype('category')

# Convertir churned de int a boolean
df['churned'] = df['churned'].astype(bool)
```

**¿Por qué?** 
- El tipo `category` usa menos memoria y es más eficiente
- El tipo `bool` es más semánticamente correcto para Sí/No

**2.3 Validación de calidad:**
```python
# Verificar duplicados
duplicados = df.duplicated().sum()  # Resultado: 0

# Verificar valores nulos
nulos = df.isnull().sum().sum()  # Resultado: 0
```

### Paso 3: Carga (Load)

El dataset transformado queda listo en memoria para el análisis y modelado.

**Resultado del ETL:**
- ✅ 5,000 registros validados
- ✅ 0 duplicados
- ✅ 0 valores nulos
- ✅ Tipos de datos optimizados
- ✅ Identificador público creado

---

## 📈 Análisis Exploratorio de Datos (EDA)

El EDA nos permite entender los datos antes de construir el modelo.

### 5.1 Distribución de la Variable Objetivo

```python
df['churned'].value_counts(normalize=True) * 100
```

| Estado | Porcentaje | Cantidad |
|--------|------------|----------|
| Churned (True) | 50.3% | 2,515 |
| Retained (False) | 49.7% | 2,485 |

**⚠️ Nota importante:** Una tasa de churn del 50% es irreal para streaming (lo típico es 2-5%). Esto indica que el dataset es **sintético/balanceado artificialmente** para entrenamiento.

### 5.2 Distribución de Variables Categóricas

| Variable | Distribución |
|----------|--------------|
| `gender` | Female: 34.2%, Male: 33.1%, Other: 32.7% |
| `subscription_type` | Premium: 33.9%, Basic: 33.2%, Standard: 32.9% |
| `region` | ~16-17% cada una (6 regiones) |
| `device` | ~19-21% cada uno (5 dispositivos) |
| `payment_method` | ~19-21% cada uno (5 métodos) |

**Observación:** Las variables categóricas están **uniformemente distribuidas**, lo cual es otra señal de que el dataset es sintético.

### 5.3 Distribución de Variables Numéricas

| Variable | Media | Desv. Est. | Mín | Máx |
|----------|-------|------------|-----|-----|
| `age` | 43.9 | 15.3 | 18 | 70 |
| `watch_hours` | 11.6 | 8.4 | 0.0 | 30.0 |
| `number_of_profiles` | 3.0 | 1.4 | 1 | 5 |
| `last_login_days` | 30.0 | 12.9 | 1 | 60 |

---

## ⚠️ Detección de Data Leakage - Problema Crítico

### ¿Qué es Data Leakage?

**Data Leakage** (fuga de datos) ocurre cuando el modelo utiliza información que **no estaría disponible** al momento de hacer predicciones en producción. Es como hacer trampa: el modelo "ve las respuestas" durante el entrenamiento.

### ¿Cómo lo detectamos?

Comparamos el promedio de cada variable entre clientes que cancelaron vs los que permanecieron:

```python
# Código de detección
for col in ['last_login_days', 'avg_watch_time_per_day', 'watch_hours', 'number_of_profiles']:
    mean_churned = df[df['churned']==True][col].mean()
    mean_retained = df[df['churned']==False][col].mean()
    ratio = mean_churned / mean_retained
    
    # Si el ratio es muy diferente de 1, hay leakage
    if ratio > 2.0 or ratio < 0.5:
        print(f"⚠️ LEAKAGE en {col}")
```

### Resultados del Análisis

| Variable | Churned | Retained | Ratio | ¿Leakage? |
|----------|---------|----------|-------|-----------|
| `last_login_days` | 38.3 días | 21.8 días | 1.76 | 🔴 **SÍ** |
| `avg_watch_time_per_day` | 0.2 hrs | 1.6 hrs | 0.10 | 🔴 **SÍ** |
| `watch_hours` | 5.9 hrs | 17.4 hrs | 0.34 | 🔴 **SÍ** |
| `number_of_profiles` | 2.8 | 3.3 | 0.86 | 🟢 **NO** |

### ¿Por qué `last_login_days` es Leakage?

**Explicación con ejemplo:**

Imagina a Juan, cliente de Netflix:
1. Juan está pensando en cancelar (pero aún no lo ha hecho)
2. Como está descontento, deja de usar Netflix
3. Pasan 40 días sin que entre a la plataforma
4. Finalmente, Juan cancela su suscripción

**El problema:** La variable `last_login_days` (40 días) es una **CONSECUENCIA** de que Juan va a cancelar, no una **CAUSA**. El cliente deja de usar la plataforma ANTES de cancelar.

**En producción:** Cuando queremos predecir si un cliente va a cancelar, no podemos saber cuántos días pasarán sin que use la app, porque eso aún no ha ocurrido.

### Analogía Simple

> Es como predecir que va a llover mirando el suelo mojado. Técnicamente aciertas, pero no sirve para decidir si llevar paraguas.
>
> **Modelo con leakage:** Mira el suelo mojado → Predice lluvia (inútil)
>
> **Modelo correcto:** Mira las nubes y humedad → Predice lluvia (útil)

### Impacto del Leakage

| Métrica | Con Leakage | Sin Leakage |
|---------|-------------|-------------|
| Accuracy | ~97% (falso) | ~77% (real) |
| Recall | ~98% (artificial) | ~86% (genuino) |
| ¿Funciona en producción? | ❌ NO | ✅ SÍ |

---

## 🎯 Selección de Variables

Basándonos en el análisis de Data Leakage, seleccionamos cuidadosamente las variables.

### Variables EXCLUIDAS (con justificación detallada)

| Variable | ¿Por qué se excluyó? |
|----------|----------------------|
| `last_login_days` | **DATA LEAKAGE** - Es consecuencia del churn, no causa. El cliente deja de usar la app ANTES de cancelar. Ratio 1.76 indica separación artificial. |
| `avg_watch_time_per_day` | **DATA LEAKAGE + Redundancia** - Ratio 0.10 (muy alejado de 1). Además, es derivable de `watch_hours`. |
| `monthly_fee` | **Variable derivada** - El precio está 100% determinado por `subscription_type`: Basic=$8.99, Standard=$13.99, Premium=$17.99. Incluir ambas genera redundancia. |
| `favorite_genre` | **Alta cardinalidad, bajo poder predictivo** - 7 categorías con distribución uniforme (~14% cada una). No correlaciona con churn. |
| `customer_id` | **Identificador único** - Solo sirve para identificar registros, no tiene valor predictivo. |
| `public_id` | **Identificador único** - Hash generado del customer_id, sin valor predictivo. |

### Variables INCLUIDAS (8 features)

| Variable | Tipo | ¿Por qué se incluyó? |
|----------|------|----------------------|
| `age` | Numérica | Factor demográfico estable. La edad puede influir en patrones de consumo y no cambia por comportamiento del cliente. |
| `watch_hours` | Numérica | Representa el **engagement total** del cliente. Es información histórica disponible antes de la predicción. |
| `number_of_profiles` | Numérica | Indica cuántas personas usan la cuenta. Más perfiles = mayor compromiso familiar = menor probabilidad de cancelar. |
| `gender` | Categórica | Factor demográfico que puede correlacionar con preferencias y permanencia. |
| `subscription_type` | Categórica | El tipo de plan (Basic, Standard, Premium) refleja el nivel de inversión del cliente en el servicio. |
| `region` | Categórica | La ubicación puede influir en disponibilidad de contenido y competencia local. |
| `payment_method` | Categórica | El método de pago puede indicar facilidad de cancelación (ej: tarjeta de crédito vs débito automático). |
| `device` | Categórica | El dispositivo puede indicar nivel de integración del servicio en la vida del cliente (TV vs móvil). |

### Código de Selección

```python
# Variables seleccionadas
X = df[["age", "gender", "subscription_type", "watch_hours", "region",
        "number_of_profiles", "payment_method", "device"]]
y = df["churned"]

print(f"✅ Variables INCLUIDAS: {list(X.columns)}")
print(f"🚫 Variables EXCLUIDAS: ['last_login_days', 'avg_watch_time_per_day', 'monthly_fee', 'favorite_genre']")
```

---

## ⚙️ Preprocesamiento de Datos

Los modelos de Machine Learning no pueden procesar texto directamente. Necesitamos transformar los datos.

### 8.1 División Train/Test

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,      # 20% para test
    random_state=42,    # Reproducibilidad
    stratify=y          # Mantener proporción de clases
)
```

| Conjunto | Registros | Porcentaje |
|----------|-----------|------------|
| Train | 4,000 | 80% |
| Test | 1,000 | 20% |

**¿Qué es `stratify=y`?** Asegura que la proporción de churned/retained sea la misma en train y test (50.3%/49.7%).

### 8.2 Transformación de Variables

**Variables Numéricas → StandardScaler**

```python
from sklearn.preprocessing import StandardScaler

# StandardScaler transforma: (valor - media) / desviación_estándar
# Resultado: media=0, desv_std=1
```

**¿Por qué?** Algunos algoritmos (como Logistic Regression) funcionan mejor cuando todas las variables numéricas están en la misma escala.

**Ejemplo:**
| Variable | Original | Escalada |
|----------|----------|----------|
| `age` | 35 años | -0.58 |
| `watch_hours` | 15.5 hrs | 0.46 |

**Variables Categóricas → OneHotEncoder**

```python
from sklearn.preprocessing import OneHotEncoder

# OneHotEncoder convierte categorías en columnas binarias (0/1)
# drop='first' elimina una categoría para evitar multicolinealidad
```

**Ejemplo con `gender`:**

| gender_Female | gender_Male | (gender_Other es la referencia) |
|---------------|-------------|--------------------------------|
| 1 | 0 | Si es Female |
| 0 | 1 | Si es Male |
| 0 | 0 | Si es Other |

**¿Qué es `drop='first'`?** Elimina la primera categoría para evitar la **"trampa de variables dummy"** (multicolinealidad). Si sabemos que no es Female ni Male, entonces es Other.

### 8.3 Pipeline de Preprocesamiento Completo

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

num_features = ["age", "watch_hours", "number_of_profiles"]
cat_features = ["gender", "subscription_type", "region", "payment_method", "device"]

preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(handle_unknown='ignore', drop='first'), cat_features)
    ]
)
```

---

## 🤖 Entrenamiento de Modelos

### 9.1 Modelos Seleccionados

Evaluamos 3 algoritmos diferentes:

| Modelo | Descripción | Ventajas |
|--------|-------------|----------|
| **Logistic Regression** | Modelo lineal que predice probabilidades | Interpretable, rápido, buen baseline |
| **Decision Tree** | Árbol de decisiones | Fácil de visualizar, captura no-linealidades |
| **Random Forest** | Conjunto de árboles | Robusto, reduce overfitting |

### 9.2 ¿Qué es Validación Cruzada?

En lugar de dividir los datos una sola vez, dividimos en **5 partes (folds)**:

```
Fold 1: [Test] [Train] [Train] [Train] [Train]
Fold 2: [Train] [Test] [Train] [Train] [Train]
Fold 3: [Train] [Train] [Test] [Train] [Train]
Fold 4: [Train] [Train] [Train] [Test] [Train]
Fold 5: [Train] [Train] [Train] [Train] [Test]
```

**¿Por qué?** Obtenemos métricas más confiables al probar en 5 conjuntos diferentes.

```python
from sklearn.model_selection import StratifiedKFold

cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

### 9.3 Búsqueda de Hiperparámetros

Los modelos tienen "perillas" llamadas **hiperparámetros** que afectan su comportamiento.

```python
from sklearn.model_selection import RandomizedSearchCV

# Ejemplo para Logistic Regression
param_logreg = {
    "model__C": np.logspace(-4, 4, 20),  # Regularización
    "model__penalty": ["l1", "l2"],       # Tipo de penalización
    "model__solver": ["liblinear"]        # Algoritmo de optimización
}

rand_search = RandomizedSearchCV(
    pipe,
    param_logreg,
    n_iter=15,           # Probar 15 combinaciones
    cv=cv_strategy,      # Validación cruzada 5-fold
    scoring=f2_scorer,   # Métrica a optimizar
    n_jobs=-1,           # Usar todos los CPUs
    random_state=42
)
```

### 9.4 ¿Por qué F2-Score como Métrica?

**Las métricas tradicionales:**

| Métrica | Fórmula | ¿Qué mide? |
|---------|---------|------------|
| **Precision** | TP / (TP + FP) | De los que predije como churn, ¿cuántos realmente eran? |
| **Recall** | TP / (TP + FN) | De los que realmente eran churn, ¿cuántos detecté? |
| **F1-Score** | 2 × (Prec × Rec) / (Prec + Rec) | Balance entre Precision y Recall |

**El problema:** F1-Score trata Precision y Recall como igualmente importantes. Pero en churn **no lo son**.

**F2-Score:** Pesa el Recall **2 veces más** que la Precision.

```python
from sklearn.metrics import fbeta_score, make_scorer

# F2-Score: beta=2 significa Recall es 2x más importante
f2_scorer = make_scorer(fbeta_score, beta=2)
```

**¿Por qué priorizar Recall?**

| Error | Nombre | Costo | Consecuencia |
|-------|--------|-------|--------------|
| No detectar churner | False Negative (FN) | **$120** | Perdemos al cliente para siempre |
| Falsa alarma | False Positive (FP) | **$10** | Solo gastamos en una oferta innecesaria |

Es **12 veces más costoso** no detectar un churner que tener una falsa alarma.

### 9.5 Mejores Hiperparámetros Encontrados

| Modelo | Mejores Parámetros |
|--------|-------------------|
| **Logistic Regression** | `C=0.00483, penalty=l2, solver=liblinear` |
| **Decision Tree** | `max_depth=3, min_samples_split=20, min_samples_leaf=8, criterion=gini` |
| **Random Forest** | `n_estimators=200, max_depth=20, min_samples_split=2, max_features=log2` |

---

## 📊 Evaluación y Resultados

### 10.1 Comparación de Modelos

| Modelo | Accuracy | Precision | Recall | F1-Score | F2-Score | ROC-AUC |
|--------|----------|-----------|--------|----------|----------|---------|
| **Logistic Regression** | **76.9%** | **73.0%** | **85.9%** | **78.9%** | **83.0%** | **85.0%** |
| Decision Tree | 73.5% | 68.8% | 86.5% | 76.7% | 82.3% | 83.2% |
| Random Forest | 76.2% | 76.5% | 76.1% | 76.3% | 76.2% | 86.5% |

**Modelo ganador: Logistic Regression** 🏆

**¿Por qué?**
- Mayor F2-Score (83.0%)
- Excelente Recall (85.9%) - detecta la mayoría de churners
- Modelo más interpretable
- Menor riesgo de overfitting

### 10.2 Matriz de Confusión

La matriz de confusión muestra los 4 resultados posibles:

```
              Predicción del Modelo
              No Churn    Churn
Realidad  No    337        160     (497 clientes leales)
          Sí     71        432     (503 clientes churners)
```

| Resultado | Cantidad | Significado |
|-----------|----------|-------------|
| **TN (337)** | True Negative | ✅ Clientes leales que predijimos como leales |
| **FP (160)** | False Positive | ⚠️ Clientes leales que predijimos como churners (falsa alarma) |
| **FN (71)** | False Negative | ❌ Churners que NO detectamos (se van sin intervención) |
| **TP (432)** | True Positive | ✅ Churners que detectamos correctamente |

### 10.3 Interpretación de Métricas

**Recall = 85.9%**
```
Recall = TP / (TP + FN) = 432 / (432 + 71) = 85.9%
```
De los 503 clientes que realmente cancelaron, detectamos 432 (85.9%).

**Precision = 73.0%**
```
Precision = TP / (TP + FP) = 432 / (432 + 160) = 73.0%
```
De los 592 clientes que predijimos como churners, 432 realmente lo eran (73.0%).

**Accuracy = 76.9%**
```
Accuracy = (TN + TP) / Total = (337 + 432) / 1000 = 76.9%
```
En general, acertamos en el 76.9% de los casos.

---

## 💰 Análisis de Impacto Financiero

### 11.1 ¿Qué es LTV (Lifetime Value)?

El **LTV (Lifetime Value)** o "Valor de Vida del Cliente" es el **total de ingresos que un cliente genera durante toda su relación con la empresa**.

**Cálculo del LTV para Netflix:**

| Concepto | Valor |
|----------|-------|
| Pago mensual promedio | $15 |
| Meses promedio de permanencia | 8 meses |
| **LTV** | **$120** ($15 × 8 meses) |

> Cada cliente que perdemos representa **$120 de ingresos futuros perdidos**.

### 11.2 Supuestos de Negocio

| Concepto | Valor | Explicación |
|----------|-------|-------------|
| **Costo FN** | $120 | LTV perdido al no detectar un churner |
| **Costo FP** | $10 | Costo de enviar oferta de retención innecesaria |
| **Beneficio TP** | $80 | LTV recuperado ($120) menos costo de retención ($40) |

**¿Por qué el beneficio TP es $80 y no $120?**
- No todos los clientes contactados aceptan quedarse (~70% de éxito)
- Gastamos dinero en la oferta de retención (descuentos, promociones)
- $120 - $40 = $80 de beneficio neto

### 11.3 Cálculo del Impacto Financiero

| Resultado | Cantidad | × | Costo/Beneficio | = | Total |
|-----------|----------|---|-----------------|---|-------|
| Churners no detectados (FN) | 71 | × | -$120 | = | **-$8,520** |
| Falsas alarmas (FP) | 160 | × | -$10 | = | **-$1,600** |
| Churners retenidos (TP) | 432 | × | +$80 | = | **+$34,560** |
| **BALANCE NETO** | | | | = | **+$24,440** |

### 11.4 Interpretación

✅ **El modelo genera +$24,440 de valor por cada 1,000 clientes evaluados**

**Proyección a escala:**

| Base de clientes | Valor generado |
|------------------|----------------|
| 1,000 clientes | +$24,440 |
| 10,000 clientes | +$244,400 |
| 100,000 clientes | +$2,444,000 |
| 1,000,000 clientes | +$24,440,000 |

---

## 🚀 Cómo Usar el Modelo

### 12.1 Requisitos

```bash
pip install pandas numpy scikit-learn joblib
```

### 12.2 Cargar y Usar el Modelo

```python
import joblib
import pandas as pd

# Cargar el modelo entrenado
model = joblib.load('churn_model_final.joblib')

# Datos de un nuevo cliente
nuevo_cliente = pd.DataFrame({
    'age': [35],
    'gender': ['Female'],
    'subscription_type': ['Premium'],
    'watch_hours': [45.5],
    'region': ['North America'],
    'number_of_profiles': [3],
    'payment_method': ['Credit Card'],
    'device': ['TV']
})

# Hacer predicción
prediccion = model.predict(nuevo_cliente)
probabilidad = model.predict_proba(nuevo_cliente)

# Mostrar resultados
print(f"Predicción: {'CHURN ⚠️' if prediccion[0] else 'NO CHURN ✅'}")
print(f"Probabilidad de churn: {probabilidad[0][1]:.1%}")
```

**Ejemplo de salida:**
```
Predicción: NO CHURN ✅
Probabilidad de churn: 15.3%
```

### 12.3 Interpretación de Resultados

| Probabilidad | Riesgo | Acción Recomendada |
|--------------|--------|-------------------|
| 0% - 30% | 🟢 Bajo | Mantener comunicación normal |
| 30% - 60% | 🟡 Medio | Monitorear, enviar encuesta de satisfacción |
| 60% - 80% | 🟠 Alto | Contactar proactivamente, ofrecer beneficios |
| 80% - 100% | 🔴 Crítico | Intervención urgente, oferta especial |

---

## 🔌 Integración con API

### 13.1 Endpoint Principal

**POST /predict**

```json
// Request
{
  "age": 35,
  "gender": "Female",
  "subscription_type": "Premium",
  "watch_hours": 45.5,
  "region": "North America",
  "number_of_profiles": 3,
  "payment_method": "Credit Card",
  "device": "TV"
}

// Response
{
  "prediction": 0,
  "probabilities": {
    "not_churn": 0.85,
    "churn": 0.15
  }
}
```

### 13.2 Otros Endpoints Disponibles

| Endpoint | Método | Descripción |
|----------|--------|-------------|
| `/predict` | POST | Predecir churn para nuevo cliente |
| `/item/predictions/{id}` | GET | Predecir para cliente existente por ID |
| `/probability/age` | GET | Probabilidad de churn por grupo de edad |
| `/probability/gender` | GET | Probabilidad de churn por género |
| `/probability/subscription` | GET | Probabilidad de churn por tipo de plan |
| `/probability/region` | GET | Probabilidad de churn por región |

### 13.3 Cambios vs Versión Anterior

| Campo | Versión Anterior | Versión Actual |
|-------|------------------|----------------|
| `last_login_days` | ✅ Requerido | ❌ **ELIMINADO** (leakage) |
| `number_of_profiles` | ❌ No incluido | ✅ **AGREGADO** |
| `payment_method` | ❌ No incluido | ✅ **AGREGADO** |
| `device` | ❌ No incluido | ✅ **AGREGADO** |

---

## 📝 Conclusiones

### 14.1 Logros del Proyecto

| Aspecto | Resultado |
|---------|-----------|
| ✅ Data Leakage | Detectado y eliminado |
| ✅ Modelo funcional | Logistic Regression con 85.9% Recall |
| ✅ Valor de negocio | +$24,440 por cada 1,000 clientes |
| ✅ Documentación | Completa y reproducible |
| ✅ Integración | Listo para API REST |

### 14.2 Comparación Final: Modelo Original vs Optimizado

| Aspecto | Original | Optimizado |
|---------|----------|------------|
| Data Leakage | ❌ Presente | ✅ Eliminado |
| Accuracy | 96.5% (falso) | 76.9% (real) |
| Recall | 98% (artificial) | 85.9% (genuino) |
| Features | 6 (con leakage) | 8 (sin leakage) |
| Validación | Simple | 5-fold estratificada |
| Funciona en producción | ❌ NO | ✅ SÍ |

### 14.3 Lecciones Aprendidas

1. **Métricas altas no siempre son buenas** - El 97% de accuracy era señal de problema, no de éxito.

2. **Entender el negocio es crucial** - Sin entender que `last_login_days` es consecuencia del churn, no habríamos detectado el leakage.

3. **F2-Score > Accuracy para churn** - Priorizar detectar churners es más valioso que accuracy general.

4. **El modelo más simple puede ser el mejor** - Logistic Regression superó a Random Forest.

---

## 📁 Estructura del Proyecto

```
ChurnInsight/
├── 📓 Rdavila_ChurnInsight_ETL_ML_Final.ipynb  # Notebook principal
├── 📦 churn_model_final.joblib                 # Modelo entrenado (Logistic Regression)
├── 📦 logreg_optimized.joblib                  # Pipeline completo LogReg
├── 📦 tree_optimized.joblib                    # Pipeline completo Decision Tree
├── 📦 rf_optimized.joblib                      # Pipeline completo Random Forest
├── 📄 README.md                                # Este archivo
└── 📊 data/
    └── netflix_churn.csv                       # Dataset original
```

---

## 👤 Autor

**R. Dávila**

- 📧 Contacto: [Tu email]
- 🔗 LinkedIn: [Tu LinkedIn]
- 🐙 GitHub: [Tu GitHub]

---

## 🙏 Agradecimientos

- **Dataset:** [Netflix Customer Churn - Kaggle](https://www.kaggle.com/)
- **Hackathon:** NoCountry
- **Equipo:** DracoStack

---

## 📄 Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para más detalles.

---

<p align="center">
  <b>⭐ Si este proyecto te fue útil, considera darle una estrella en GitHub ⭐</b>
</p>
