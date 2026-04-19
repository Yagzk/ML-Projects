# ⚾ Baseball Salary — Ensemble Methods Comparison (XGBoost, LightGBM, CatBoost & More)

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![Algorithm](https://img.shields.io/badge/Algorithm-Ensemble%20Methods-orange) ![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

## 📌 Overview

A comprehensive **ensemble methods benchmark** project predicting Major League Baseball player salaries. Every major regression algorithm is trained, tuned with GridSearchCV, and compared — from simple Linear Regression all the way to CatBoost. Feature importance analysis is performed using LightGBM.

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| Source | Hitters Dataset (ISLR) |
| File | `Hitters.csv` |
| Rows | 322 players |
| Target | `Salary` (annual salary in thousands USD) |
| Task | Regression |

**Features:** At-bats, hits, home runs, runs, RBIs, walks, years, career stats, league, division, new league.

---

## 🔧 Full Pipeline

### 1. EDA & Missing Values
- `Salary` had missing values → filled with **median**
- Distribution was right-skewed — many low earners, few stars
- Correlation heatmap revealed strong relationships among career stats (CHits, CAtBat, etc.)

### 2. Encoding
- `League`, `Division`, `NewLeague` → **One-Hot Encoding** (drop_first=True)

### 3. Outlier Detection
```python
for col in numerical_cols:
    z_scores = np.abs(stats.zscore(df[col]))
    outlier_count = (z_scores > 3).sum()
```
Visualized with boxplots per column.

### 4. Outlier Suppression (Winsorization/Clipping)
Values beyond 3 standard deviations were capped rather than removed, to preserve dataset size.

> ⚠️ **Important finding:** After outlier suppression, Linear, Ridge, and Lasso models started **overfitting** — the clipping introduced new patterns that simpler models couldn't generalize. Ensemble methods remained robust.

### 5. Standardization
```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
```

---

## 🤖 Models Trained

All models were tuned with `GridSearchCV(cv=5, scoring='r2')`:

### Linear Models
| Model | Params | Notes |
|-------|--------|-------|
| Linear Regression | — | Baseline, R²=0.36 after outlier clipping |
| Ridge | `alpha: logspace(-3,3,50)` | Overfit after clipping |
| Lasso | `alpha: logspace(-3,1,50)` | Feature selection via L1 |

### Ensemble / Tree Models
| Model | Key Params | Strengths |
|-------|-----------|-----------|
| **Random Forest** | n_estimators, max_depth, min_samples | Robust, low variance |
| **Gradient Boosting** | n_estimators, learning_rate, max_depth | Strong performance |
| **XGBoost** | n_estimators, learning_rate, max_depth | Fast, regularized boosting |
| **LightGBM** | n_estimators, learning_rate, max_depth | Fastest training, great for large data |
| **CatBoost** | iterations, learning_rate, depth | Handles categorical natively |

### GridSearchCV Example
```python
xgb_params = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1],
    'max_depth': [3, 5, 10]
}
xgb_grid = GridSearchCV(XGBRegressor(random_state=42), xgb_params, scoring='r2', cv=5)
xgb_grid.fit(X_train, y_train)
```

---

## 📊 Feature Importance — LightGBM

```python
importances = best_lgbm.feature_importances_
importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
importance_df.sort_values('Importance', ascending=False)
```
Top performing features were career statistics (CHits, CAtBat, CRuns) — accumulated performance over a player's career outweighs a single season's stats.

---

## 📉 Results Summary

| Model | Behavior |
|-------|---------|
| Linear / Ridge / Lasso | Overfit after outlier clipping |
| Random Forest | Robust, good generalization |
| Gradient Boosting | Strong CV score |
| XGBoost | Competitive, fast |
| **LightGBM** | Best training speed + competitive accuracy |
| CatBoost | Excellent, especially for categorical features |

> **Conclusion:** Ensemble methods (especially gradient boosting variants) significantly outperform linear models for this dataset. Outlier clipping hurts linear models more than tree-based ones.

---

## 📦 Libraries Used

```python
pandas, numpy, scipy (stats), sklearn (LinearRegression, Ridge, Lasso, RandomForestRegressor,
GradientBoostingRegressor, GridSearchCV, StandardScaler, cross_validate),
xgboost (XGBRegressor), lightgbm (LGBMRegressor), catboost (CatBoostRegressor),
seaborn, matplotlib
```

---

## 💡 Key Takeaways

1. **Outlier handling strategy matters** — clipping helped ensemble models but hurt linear ones
2. **Career stats > season stats** for salary prediction (teams pay for track record)
3. LightGBM provides the best speed/accuracy trade-off for tabular regression
4. Always compare linear baselines against ensemble methods — the gap tells you how much non-linearity is in the data
5. `cross_validate` instead of a simple train/test gives a much more honest estimate of model quality
