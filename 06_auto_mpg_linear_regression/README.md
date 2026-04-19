# 🚗 Auto MPG — Linear Regression + VIF + Ridge

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![Algorithm](https://img.shields.io/badge/Algorithm-Linear%20%7C%20Ridge%20Regression-orange) ![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

## 📌 Overview

Predicting vehicle **fuel efficiency (MPG — miles per gallon)** using the UCI Auto MPG dataset. The core focus is diagnosing and resolving **multicollinearity** using VIF (Variance Inflation Factor) analysis, applying feature engineering, and comparing Linear Regression with Ridge. The final model is serialized with joblib.

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| Source | UCI Machine Learning Repository (`fetch_ucirepo(id=9)`) |
| Rows | 392 vehicles |
| Target | `mpg` — miles per gallon (continuous) |
| Task | Regression |

**Features:**
| Column | Description |
|--------|-------------|
| `cylinders` | Number of engine cylinders |
| `displacement` | Engine displacement (cubic inches) |
| `horsepower` | Engine horsepower |
| `weight` | Vehicle weight (pounds) |
| `acceleration` | 0–60 mph time (seconds) |
| `model_year` | Model year |
| `origin` | Country of origin (1=USA, 2=Europe, 3=Japan) |

---

## 🔧 Methodology

### 1. Data Cleaning
```python
X['horsepower'] = X['horsepower'].fillna(X['horsepower'].median())
X = pd.get_dummies(X, columns=["origin"], drop_first=True)  # Nominal → One-Hot
```

### 2. Train/Test Split → Then Scale
```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

continuous = ["displacement", "horsepower", "weight", "acceleration"]
scaler = StandardScaler()
X_train[continuous] = scaler.fit_transform(X_train[continuous])
X_test[continuous]  = scaler.transform(X_test[continuous])
```

### 3. Baseline Linear Regression
```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

lr_model = LinearRegression()
lr_model.fit(X_train[continuous], y_train)
y_pred = lr_model.predict(X_test[continuous])

print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")
print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")
print(f"R²:  {r2_score(y_test, y_pred):.4f}")
```

### 4. VIF Analysis — Multicollinearity Detection
```python
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif_data = pd.DataFrame()
vif_data['Feature'] = df_local.columns
vif_data['VIF'] = [variance_inflation_factor(df_local.values, i)
                   for i in range(df_local.shape[1])]
```

**High VIF detected:**
- `displacement`, `cylinders`, `weight`, `horsepower` → very high VIF (>10)

> **Note:** `model_year` consistently inflated VIF regardless of any transformations applied — it had to be excluded.

### 5. Feature Engineering to Reduce VIF
```python
X["displacement_per_cyl"]    = X["displacement"] / X["cylinders"]
X["acceleration_per_weight"] = X["acceleration"] / X["weight"]

# Original high-VIF columns dropped
df_dropped = df_local.drop(columns=['horsepower','weight','displacement',
                                     'acceleration','model_year','cylinders'])
```

### 6. Ridge Regression (Final Model)
```python
from sklearn.linear_model import Ridge

X_train, X_test, y_train, y_test = train_test_split(df_dropped, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train_scaled, y_train)
y_pred = ridge_model.predict(X_test_scaled)

print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")
print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")
print(f"R²:  {r2_score(y_test, y_pred):.4f}")
```

> **Finding:** Higher displacement-per-cylinder reduces MPG (heavier engine → more fuel). Higher acceleration-per-weight increases MPG (efficient power-to-weight ratio). Coefficient directions align with real-world physics.

### 7. Model Serialization
```python
from joblib import dump
dump(lr_model, 'mpg.joblib')
```

---

## 📦 Libraries Used
```python
pandas, numpy, matplotlib, seaborn,
ucimlrepo (fetch_ucirepo),
sklearn (LinearRegression, Ridge, StandardScaler, train_test_split,
mean_absolute_error, mean_squared_error, r2_score),
statsmodels (variance_inflation_factor),
joblib
```

---

## 💡 Key Takeaways
- **Multicollinearity** doesn't ruin predictions but destroys coefficient interpretability — VIF > 10 is serious
- Feature engineering (`displacement / cylinders`) is more powerful than simply dropping features
- `model_year` was impossible to include without inflating VIF — domain constraints sometimes override intuition
- One-Hot Encoding for `origin` is essential — treating USA=1, Europe=2, Japan=3 implies ordering that doesn't exist
- Ridge was applied after feature engineering to additionally penalize remaining multicollinearity
- Always split data **before** scaling — fitting scaler on full data leaks test information
