# ⚾ Baseball Salary — Ridge, Lasso & Polynomial Regression

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![Algorithm](https://img.shields.io/badge/Algorithm-Ridge%20%7C%20Lasso%20%7C%20Pipeline-orange) ![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

## 📌 Overview

Comprehensive regularization study predicting Major League Baseball player salaries. Compares Linear Regression, Ridge, Lasso — both with and without Polynomial Features — and uses GridSearchCV with cross-validation to find optimal alpha values. Also demonstrates sklearn Pipeline to prevent data leakage in cross-validation.

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| Source | Hitters Dataset (ISLR) |
| File | `hitters_processed.pkl` |
| Target | `Salary` (player annual salary in thousands USD) |
| Task | Regression |

**Features:** AtBat, Hits, HmRun, Runs, RBI, Walks, Years, career stats (CHits, CAtBat...), League, Division, NewLeague

---

## 🔧 Methodology

### 1. Data Preparation
```python
X = data.drop(columns=['Salary'])
y = data['Salary']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
```

### 2. Baseline — Linear Regression
```python
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)

print(f"MSE: {mean_squared_error(y_test, y_pred_lr):.2f}")
print(f"R²:  {r2_score(y_test, y_pred_lr):.2f}")
```
> **Finding:** Scatter plot shows predictions deviating significantly from the x=y line — regularization needed.

### 3. Ridge — Sabit ve Otomatik Alpha
```python
# Fixed alpha
ridge = Ridge(alpha=20)
ridge.fit(X_train_scaled, y_train)

# Automated with GridSearchCV
ridge_params = {'alpha': np.logspace(-3, 3, 50)}
ridge_grid = GridSearchCV(Ridge(), ridge_params, scoring='r2', cv=5)
ridge_grid.fit(X_train_scaled, y_train)
best_ridge = ridge_grid.best_estimator_
```

### 4. Lasso — Sabit ve Otomatik Alpha
```python
# Fixed alpha
lasso = Lasso(alpha=20, max_iter=10000)

# Automated with GridSearchCV
lasso_params = {'alpha': np.logspace(-3, 0, 50)}
lasso_grid = GridSearchCV(Lasso(max_iter=10000), lasso_params, scoring='r2', cv=5)
lasso_grid.fit(X_train_scaled, y_train)
```

### 5. Alpha vs MSE Visualization
```python
# MSE computed per alpha and visualized
# Ridge lowest MSE: alpha ≈ 10⁻³
# Lasso lowest MSE: alpha between 10⁻¹ and 1
```

### 6. Polynomial Features Denemesi
```python
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train_c)
X_test_poly  = poly.transform(X_test_c)

# Polinom + Standardizasyon
scaler = StandardScaler()
X_train_poly_scaled = scaler.fit_transform(X_train_poly)
X_test_poly_scaled  = scaler.transform(X_test_poly)

lr_poly = LinearRegression()
lr_poly.fit(X_train_poly_scaled, y_train_c)
```
> ⚠️ **Critical finding:** Adding polynomial features resulted in negative R². Too many features + multicollinearity → overfitting. Ridge and Lasso with polynomial features were also tested — they also produced poor R² values.

### 7. Pipeline (Data Leakage Önleme)
```python
from sklearn.pipeline import Pipeline

pipeline_ridge = Pipeline([
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('scaler', StandardScaler()),
    ('ridge', Ridge())
])

param_grid = {'ridge__alpha': np.logspace(-3, 3, 50)}
grid = GridSearchCV(pipeline_ridge, param_grid, scoring='r2', cv=5)
grid.fit(X_train, y_train)
```
> Pipeline fits each preprocessing step separately per fold during cross-validation, preventing data leakage.

---

## 📊 Model Karşılaştırması

| Model | Notes |
|-------|--------|
| Linear Regression | Baseline; scatter plot shows significant deviation |
| Ridge (alpha=20) | Manual, suboptimal |
| Ridge (GridSearchCV) | Best alpha found via cross-validation |
| Lasso (alpha=20) | Aggressive shrinkage |
| Lasso (GridSearchCV) | Automatic feature selection |
| Linear + Poly(degree=2) | R² negative → overfit |
| Ridge/Lasso + Poly | Low R² → polynomial not suitable for this dataset |
| Pipeline | Safest approach |

---

## 📦 Libraries Used
```python
pandas, numpy, matplotlib,
sklearn (LinearRegression, Ridge, Lasso, PolynomialFeatures,
GridSearchCV, Pipeline, StandardScaler, train_test_split,
mean_squared_error, r2_score)
```

---

## 💡 Key Takeaways
- Ridge shrinks all coefficients but keeps all features — L2 regularization
- Lasso eliminates some coefficients entirely — L1 regularization = built-in feature selection
- Polynomial features on correlated sports stats → overfitting (R² < 0) — feature space explodes
- **Pipeline** prevents data leakage in cross-validation — always use it with preprocessing + CV
- `np.logspace(-3, 3, 50)` provides a logarithmically spaced search range — better than linear for alpha
- GridSearchCV result (best alpha) may differ from the manual MSE plot because CV uses validation splits
