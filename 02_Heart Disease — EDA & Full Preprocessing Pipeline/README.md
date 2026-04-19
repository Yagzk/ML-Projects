# 🫀 Heart Disease — EDA & Full Preprocessing Pipeline

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

## 📌 Overview

Two-notebook progression on the UCI Heart Disease dataset. **Hearth_disease_1** is the first contact with the data — exploration, outlier analysis, and strategy decisions. **hearth_disease_2** builds the complete preprocessing pipeline. Together they show the natural learning arc from raw data discovery to a clean, model-ready dataset.

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| Source | UCI Machine Learning Repository |
| Rows | 920 patients |
| Columns | 16 features |
| Target | `num` (0 = no disease, 1–4 = severity) |
| Sources | Cleveland, Hungary, Switzerland, VA Long Beach |

**Key features:** `age`, `sex`, `cp` (chest pain), `trestbps` (resting BP), `chol` (cholesterol), `thalch` (max heart rate), `oldpeak` (ST depression), `ca` (vessels colored), `thal`, `exang`

---

## 📂 Notebooks

### `Hearth_disease_1.ipynb` — Exploration + Outlier Analysis

```python
data.tail()
data.dtypes
data.shape           # (920, 16)
data.describe()
data.isnull().sum()
```

**Missing value strategies explored:**
```python
# Too destructive — drops useful columns
data_dropped_columns = data.dropna(axis=1)

# Selective column drop
data.drop(columns=["ca", "thal"], inplace=True)

# SimpleImputer with median
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')
data['thalch'] = imputer.fit_transform(data[['thalch']])
```

**IQR Outlier Analysis (example on chol column):**
```python
Q1 = df['chol'].quantile(0.25)
Q3 = df['chol'].quantile(0.75)
IQR = Q3 - Q1
lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR
outliers = df[(df['chol'] < lower_limit) | (df['chol'] > upper_limit)]
```

**Systematic missing value fill (all columns):**
```python
# Numerical → mean
for col in numeric_columns:
    if data[col].isnull().sum() > 0:
        data[col].fillna(data[col].mean(), inplace=True)

# Categorical → mode
for col in categoric_columns:
    if data[col].isnull().sum() > 0:
        data[col].fillna(data[col].mode()[0], inplace=True)
```

A **boxplot + IQR** outlier analysis was performed for every numerical column.

---

### `hearth_disease_2.ipynb` — Full Preprocessing Pipeline

#### 1. Missing Value Handling
```python
# Categorical → mode
for column in categorical_columns:
    df[column].fillna(df[column].mode()[0], inplace=True)

# Numerical → median
for column in numeric_columns:
    df[column].fillna(df[column].median(), inplace=True)
```

#### 2. Encoding
| Column | Method | Reason |
|--------|--------|--------|
| `sex`, `fbs`, `exang` | Label Encoding | Binary categories |
| `dataset`, `cp`, `restecg`, `slope`, `thal` | One-Hot Encoding | Nominal, no ordinal relationship |

```python
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
df['sex']   = label_encoder.fit_transform(df['sex'])
df['fbs']   = label_encoder.fit_transform(df['fbs'])
df['exang'] = label_encoder.fit_transform(df['exang'])

one_hot = pd.get_dummies(df[['dataset','cp','restecg','slope','thal']], drop_first=True)
df = pd.concat([df, one_hot], axis=1)
df.drop(['dataset','cp','restecg','slope','thal'], axis=1, inplace=True)
```

#### 3. Standardization
```python
from sklearn.preprocessing import StandardScaler

target_column = 'num'
numeric_columns = numeric_columns.drop(target_column)
scaler = StandardScaler()
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
# Result: mean ≈ 0, std ≈ 1 for all numerical features
```

#### 4. Output
```python
df.to_csv("heart_disease_sonn.csv", index=False)
```

---

## 📦 Libraries Used
```python
pandas, numpy, matplotlib, scipy (stats),
sklearn.impute (SimpleImputer),
sklearn.preprocessing (LabelEncoder, StandardScaler)
```

---

## 💡 Key Takeaways
- `ca` (611 missing) and `thal` (486 missing) — dropna would destroy too many useful features
- `dropna(axis=1)` removes whole columns — explored but rejected due to information loss
- IQR outlier analysis done per column with boxplots — systematic, not arbitrary
- One-Hot Encoding for `thal` avoids false ordinal: fixed=0, normal=1, reversable=2 would imply a medical ordering that doesn't exist
- StandardScaler applied **after** encoding — correct order prevents data leakage
