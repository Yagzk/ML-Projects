# 🏥 Insurance Cost — Polynomial Regression, Log Transform & Regularization

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![Algorithm](https://img.shields.io/badge/Algorithm-Polynomial%20%7C%20Ridge%20%7C%20Lasso-orange) ![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

## 📌 Overview

Comprehensive regression study predicting annual medical insurance charges. This project explores multiple approaches: linear regression with interaction features, polynomial regression (degree search), log-transformed target, and Ridge/Lasso regularization. The key discovery: domain-informed interaction features make polynomial transformation unnecessary.

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| File | `insurance.csv` |
| Rows | 1,338 records |
| Target | `charges` (annual insurance cost in USD) |
| Task | Regression |

**Features:** `age`, `sex`, `bmi`, `children`, `smoker`, `region`

---

## 🔧 Methodology

### 1. Preprocessing + Interaction Features
```python
# Sayısal → StandardScaler
numerical_features = ['age', 'bmi', 'children']
scaler = StandardScaler()
data[numerical_features] = scaler.fit_transform(data[numerical_features])

# Kategorik → One-Hot Encoding
data = pd.get_dummies(data, columns=['sex', 'smoker', 'region'], drop_first=True)

# Interaction features
data['smoker_yes_age'] = data['smoker_yes'] * data['age']
data['smoker_yes_bmi'] = data['smoker_yes'] * data['bmi']
```

### 2. Correlation Analysis
Heatmap → `smoker_yes` ↔ `charges` shows the strongest correlation.
> *"There is a strong relationship between smoking status and charges. Seriously, don't smoke."*

### 3. Linear Regression (Baseline)
```python
lin_model = LinearRegression()
lin_model.fit(X_train, y_train)
y_pred_lin = lin_model.predict(X_test)

mse_lin = mean_squared_error(y_test, y_pred_lin)
r2_lin  = r2_score(y_test, y_pred_lin)
print(f"Lineer Regresyon: MSE={mse_lin:.2f}, R²={r2_lin:.2f}")
```
> Thanks to feature engineering, R² turned out to be quite high.

### 4. Log Transform Denemesi
```python
data['log_charges'] = np.log(data['charges'])
y_log = data['log_charges']

lr_log = LinearRegression()
lr_log.fit(X_loglu_train, y_train_log)
y_pred_log = lr_log.predict(X_loglu_test)
```
> **Finding:** Log transform normalized the distribution, but the non-log model achieved higher R². Both approaches were compared.

### 5. Polynomial Degree Search (Loop)
```python
degrees = range(1, 5)
best_degree = None
best_mse = float("inf")

for degree in degrees:
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly  = poly.transform(X_test)

    lr_poly = LinearRegression()
    lr_poly.fit(X_train_poly, y_train)
    y_pred_poly = lr_poly.predict(X_test_poly)

    mse_poly = mean_squared_error(y_test, y_pred_poly)
    r2_poly  = r2_score(y_test, y_pred_poly)
```

**Without log transform:** Optimal degree = **1** (directly linear)
**With log-transformed target:** Optimal degree = **2**

> **Key finding:** After feature engineering, polynomial degree 1 is optimal — the interaction features already captured the non-linearity. Increasing degree only increases overfitting risk.

### 6. Ridge & Lasso (Manuel Alpha)
```python
# Lasso
lasso = Lasso(alpha=20, max_iter=10000)
lasso.fit(X_train, y_train)

# Ridge
ridge = Ridge(alpha=20)
ridge.fit(X_train, y_train)
```
> Manual alpha already yields high R² — the effect of feature engineering is visible here too.

### 7. Ridge & Lasso (GridSearchCV + Log target)
```python
# Ridge with log target
ridge_params = {'alpha': np.logspace(0.001, 100, 100)}
ridge_grid = GridSearchCV(Ridge(), ridge_params, scoring='r2', cv=5)
ridge_grid.fit(X_loglu_train, y_train_log)

# Lasso with log target
lasso_params = {'alpha': np.logspace(-3, 2, 100)}
lasso_grid = GridSearchCV(Lasso(max_iter=10000), lasso_params, scoring='r2', cv=5)
lasso_grid.fit(X_loglu_train, y_train_log)
```
> Ridge/Lasso with log target: MSE decreased but R² also decreased. After the exponential inverse transform, R² dropped further — log transform had a negative effect for this dataset.

---

## 📊 Model Karşılaştırması

| Model | Notes |
|-------|--------|
| Linear (with feature engineering) | Best R² — simple and powerful |
| Linear (log target) | R² remained lower |
| Polynomial degree=1 | Non-log veride optimal |
| Polynomial degree=2 | Log veride optimal, non-log veride gereksiz |
| Lasso / Ridge (alpha=20) | High R² — feature engineering effect |
| Ridge + log + GridSearchCV | MSE ↓ but R² ↓ — log was negative for this dataset |

---

## 📦 Libraries Used
```python
pandas, numpy, seaborn, matplotlib,
sklearn (LinearRegression, Ridge, Lasso, PolynomialFeatures,
GridSearchCV, StandardScaler, train_test_split,
mean_squared_error, r2_score)
```

---

## 💡 Key Takeaways
- **Smoking is the dominant predictor** — a single binary variable reshapes the entire cost distribution
- `smoker_yes × age` and `smoker_yes × bmi` interaction features eliminate need for polynomial complexity
- Log transform normalizes distribution but doesn't always improve R² — always compare empirically
- Polynomial degree should be determined with a loop, not guessed — degree=1 won here
- Ridge and Lasso with manual alpha already perform well; GridSearchCV only helps marginally when feature engineering is strong
