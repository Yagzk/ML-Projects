# ⚾ Baseball Salary Prediction — Ridge & Lasso Regularization

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![Algorithm](https://img.shields.io/badge/Algorithm-Ridge%20%7C%20Lasso-orange) ![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

## 📌 Overview

Predicting **Major League Baseball player salaries** using Ridge and Lasso regularization. This project compares the performance of standard Linear Regression against regularized models, and uses `GridSearchCV` with cross-validation to automatically find the optimal regularization strength (alpha).

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| Source | Hitters dataset (ISLR) |
| File | `hitters_processed.pkl` |
| Task | Regression (salary prediction) |
| Target | `Salary` (player annual salary in thousands USD) |

**Features include:** At-bats, hits, home runs, runs, RBIs, walks, years played, career stats (CHits, CAtBat, etc.), league, division.

---

## 🔧 Methodology

### 1. Train/Test Split + Standardization
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
```

### 2. Baseline — Linear Regression
- Trained and evaluated on scaled data
- Scatter plot: predicted vs actual values
- **Observation:** Some predictions deviated significantly from the x=y line, indicating room for improvement

### 3. Ridge Regression — L2 Regularization

**Fixed alpha first (α=20):**
```python
ridge = Ridge(alpha=20)
```

**Then automated with GridSearchCV:**
```python
ridge_params = {'alpha': np.logspace(-3, 3, 50)}
ridge_grid = GridSearchCV(ridge, ridge_params, scoring='r2', cv=5)
```
Ridge shrinks all coefficients toward zero but keeps all features — good when all features contribute something.

### 4. Lasso Regression — L1 Regularization

**Fixed alpha first (α=20):**
```python
lasso = Lasso(alpha=20, max_iter=10000)
```

**Then automated with GridSearchCV:**
```python
lasso_params = {'alpha': np.logspace(-3, 0, 50)}
lasso_grid = GridSearchCV(lasso, lasso_params, scoring='r2', cv=5)
```
Lasso performs **automatic feature selection** by shrinking some coefficients to exactly zero — useful for identifying the most important predictors.

---

## 📉 Model Comparison

| Model | Alpha | Notes |
|-------|-------|-------|
| Linear Regression | — | Baseline, no regularization |
| Ridge (fixed) | 20 | Manually set, suboptimal |
| Ridge (GridSearchCV) | Auto (cv=5) | Best alpha via cross-validation |
| Lasso (fixed) | 20 | Aggressive shrinkage |
| Lasso (GridSearchCV) | Auto (cv=5) | Automatic feature selection |

---

## 📦 Libraries Used

```python
pandas, numpy, sklearn (LinearRegression, Ridge, Lasso, GridSearchCV, StandardScaler,
train_test_split), matplotlib
```

---

## 💡 Key Takeaways

- Regularization is essential when features are correlated (as in sports statistics)
- **Ridge** is preferred when all features are believed to be relevant — it shrinks but doesn't eliminate
- **Lasso** is preferred for feature selection — it eliminates irrelevant predictors entirely
- Always tune alpha with `GridSearchCV` instead of guessing — `np.logspace(-3, 3, 50)` provides a good search range
- Cross-validation (cv=5) ensures the best alpha generalizes to unseen data
