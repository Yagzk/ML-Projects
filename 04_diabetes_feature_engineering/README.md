# 🩺 Diabetes Prediction — Feature Engineering & Preprocessing

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

## 📌 Overview

This project applies advanced **feature engineering and preprocessing** techniques to the Pima Indians Diabetes dataset. The focus is on creating meaningful new features, handling biologically impossible zero values, and preparing a clean, model-ready dataset for binary diabetes classification.

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| Source | Pima Indians Diabetes Database |
| Rows | 768 patients |
| Target | `Outcome` (1 = Diabetic, 0 = Non-Diabetic) |

**Features:**
| Column | Description |
|--------|-------------|
| `Pregnancies` | Number of pregnancies |
| `Glucose` | Plasma glucose concentration |
| `BloodPressure` | Diastolic blood pressure (mm Hg) |
| `SkinThickness` | Triceps skin fold thickness (mm) |
| `Insulin` | 2-Hour serum insulin (mu U/ml) |
| `BMI` | Body Mass Index |
| `DiabetesPedigreeFunction` | Diabetes family history score |
| `Age` | Age in years |

---

## 🔧 What Was Done

### 1. Zero Value Treatment
Biologically impossible zero values in `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI` were replaced with **column median** (not dropped, to preserve data):
```python
for col in missing_cols:
    df[col] = df[col].replace(0, df[col].median())
```

### 2. Feature Engineering — New Columns Created

| Feature | Formula | Rationale |
|---------|---------|-----------|
| `AgeCategory` | `pd.cut(Age, bins=[0,12,18,35,55,100])` | Captures non-linear age effects (Baby/Child/Young/Middle-Aged/Elderly) |
| `Pregnancy_Age_Ratio` | `Pregnancies / Age` | Relative pregnancy burden by age |
| `HealthRiskScore` | `BMI × Glucose` | Combined metabolic risk indicator |

> **Result:** `HealthRiskScore` showed the strongest correlation with the `Outcome` target variable

### 3. Encoding
- `AgeCategory` → **Label Encoding** (ordinal category)

### 4. Standardization
- Applied `StandardScaler` to all numerical columns including engineered features:
```python
numerical_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                     'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age',
                     'Pregnancy_Age_Ratio', 'HealthRiskScore']
```

### 5. Correlation Analysis
```
HealthRiskScore          → Highest correlation with Outcome
Glucose                  → 2nd strongest predictor
BMI                      → Moderate correlation
DiabetesPedigreeFunction → Moderate correlation
```

---

## 📦 Libraries Used

```python
pandas, numpy, sklearn.preprocessing (StandardScaler, LabelEncoder), seaborn, matplotlib
```

---

## 💡 Key Takeaways

- Zero values in medical datasets are often missing values in disguise — domain knowledge is essential
- Combining `BMI × Glucose` into a single health risk score outperforms either feature alone
- Age binning helps capture non-linear age effects that a raw numeric column might miss
- Feature engineering can significantly boost model performance before even selecting an algorithm
