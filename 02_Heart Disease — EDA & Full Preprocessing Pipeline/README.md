# 🫀 Heart Disease EDA & Preprocessing — Full Pipeline (v1 → v2)

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

## 📌 Overview

Two-notebook progression on the UCI Heart Disease dataset — from **first exploration (v1)** to a **complete preprocessing pipeline (v2)**. Shows the natural learning arc: initial data discovery → systematic cleaning → encoding → scaling → export.

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| Source | UCI Machine Learning Repository |
| Rows | 920 patients |
| Columns | 16 features |
| Target | `num` (0 = no disease, 1–4 = disease severity) |
| Hospital Sources | Cleveland, Hungary, Switzerland, VA Long Beach |

**Key features:** `age`, `sex`, `cp` (chest pain type), `trestbps` (resting blood pressure), `chol` (cholesterol), `thalch` (max heart rate), `oldpeak` (ST depression), `ca` (vessels colored), `thal` (thalassemia type), `exang` (exercise angina)

---

## 📂 Notebooks

### v1 — `Hearth_disease_1.ipynb` · First Exploration
The initial encounter with the dataset. No cleaning applied yet — purely understanding the structure.

```python
data.tail()            # Last rows
data.dtypes            # Column types
data.shape             # (920, 16)
data.describe()        # Statistical summary
data.isnull().sum()    # Missing value counts
```

**Missing value strategies explored (not yet applied):**
- `dropna(axis=1)` — drops all columns with any NaN → too destructive, loses useful columns
- `dropna(axis=0)` — drops rows with NaN → loses too many records
- `fillna(median)` / `fillna(mode)` — imputation → preferred approach

---

### v2 — `Hearth_disease_2.ipynb` · Full Preprocessing Pipeline
The complete pipeline built on top of the v1 exploration.

#### 1. Missing Value Handling
- **Numerical columns** → filled with **median** (robust to outliers)
- **Categorical columns** → filled with **mode** (most frequent value)

#### 2. Encoding
| Column | Method | Reason |
|--------|--------|--------|
| `sex`, `fbs`, `exang` | Label Encoding | Binary categories |
| `dataset`, `cp`, `restecg`, `slope`, `thal` | One-Hot Encoding | Nominal multi-class, no ordinal relationship |

#### 3. Standardization
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
# Result: mean ≈ 0, std ≈ 1 for all numerical features
```

#### 4. Output
- Cleaned dataset exported as `heart_disease_sonn.csv`

---

## 📦 Libraries Used

```python
pandas, numpy, sklearn.preprocessing (LabelEncoder, OneHotEncoder, StandardScaler)
```

---

## 💡 Key Takeaways

- `ca` (611 missing) and `thal` (486 missing) had >50% missing values — median/mode imputation was the only viable option
- `dropna(axis=1)` is tempting but destructive — it would remove nearly half the feature columns
- One-Hot Encoding for multi-class nominals avoids false ordinal relationships (e.g. thal: fixed=0, normal=1, reversable=2 would imply an ordering that doesn't exist medically)
- StandardScaler must be applied **after** encoding, **after** train/test split in a full ML pipeline to prevent data leakage
