# ⚾ Baseball Salary — Ensemble Boosting Regression

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![Algorithm](https://img.shields.io/badge/Algorithm-XGBoost%20%7C%20LightGBM%20%7C%20CatBoost%20%7C%20GBM%20%7C%20RF-orange) ![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

## 📌 Overview

Comprehensive ensemble regression benchmark predicting MLB player salaries. Every major algorithm is trained and tuned with GridSearchCV: Linear, Ridge, Lasso, Random Forest, Gradient Boosting, XGBoost, LightGBM, and CatBoost. Includes a critical finding about outlier suppression: it caused overfitting in linear models but ensemble models remained robust.

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| File | `Hitters.csv` |
| Rows | 322 players |
| Target | `Salary` (annual salary in thousands USD) |
| Task | Regression |

**Features:** AtBat, Hits, HmRun, Runs, RBI, Walks, Years, career stats (CHits, CAtBat, CRuns, CRBI, CWalks), PutOuts, Assists, Errors, League, Division, NewLeague

---

## 🔧 Methodology

### 1. EDA
```python
plt.hist(df["Salary"], bins=30)
# Maaşlar düşük değerde yığılmış — sağa çarpık dağılım
```

### 2. Missing Values + Encoding
```python
df["Salary"] = df["Salary"].fillna(df["Salary"].median())

# League, Division, NewLeague → One-Hot Encoding
df = pd.get_dummies(df, columns=["League","Division","NewLeague"], drop_first=True)
```

### 3. Correlation Analysis
```python
corr_matrix = df[num_cols].corr()
salary_corr = df[num_cols].corrwith(df["Salary"]).sort_values(ascending=False)
# Kariyer istatistikleri (CHits, CAtBat) tek sezon istatistiklerinden daha korelasyonlu
```

### 4. Outlier Detection (Z-Score, per column)
```python
from scipy import stats

for col in numerical_cols:
    z_scores = np.abs(stats.zscore(df[col]))
    outlier_count = (z_scores > 3).sum()
    # Boxplot visualization + outlier suppression (clipping)
```

### 5. Train/Test Split + Scaling
```python
X = df.drop("Salary", axis=1)
y = df["Salary"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
```

### 6. Linear Regression
```python
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)

# Cross-validation
cv_results = cross_validate(lr_model, X_train_scaled, y_train, cv=5,
                             scoring=['r2','neg_mean_absolute_error','neg_root_mean_squared_error'])
```
**Results (Suppressed Data):**
```
MAE:  233.11
MSE:  122555.00
R²:   0.36
CV R²: 0.29  ← Cross-validation R² very low → OVERFITTING
```

### 7. Ridge Regression (GridSearchCV)
```python
ridge_params = {'alpha': np.logspace(-3, 3, 50)}
ridge_grid = GridSearchCV(Ridge(), ridge_params, scoring='r2', cv=5)
ridge_grid.fit(X_train_clip_scaled, y_train_clip)
```
> After outlier suppression, Ridge also overfit — cross-validation R² dropped.

### 8. Lasso Regression (GridSearchCV)
```python
lasso_params = {'alpha': np.logspace(-3, 1, 50)}
lasso_grid = GridSearchCV(Lasso(max_iter=10000), lasso_params, scoring='r2', cv=5)
```

### 9. Random Forest (GridSearchCV)
```python
rf_params = {
    'n_estimators': [100, 200],
    'max_depth':    [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf':  [1, 2]
}
rf_grid = GridSearchCV(RandomForestRegressor(random_state=42),
                       rf_params, scoring='r2', cv=5, n_jobs=-1)
rf_grid.fit(X_train, y_train)
```

### 10. Gradient Boosting (GridSearchCV)
```python
gb_params = {'n_estimators':[100,200], 'learning_rate':[0.01,0.1], 'max_depth':[3,5,10]}
gb_grid = GridSearchCV(GradientBoostingRegressor(random_state=42),
                       gb_params, scoring='r2', cv=5, n_jobs=-1)
```

### 11. XGBoost (GridSearchCV)
```python
xgb_params = {'n_estimators':[100,200], 'learning_rate':[0.01,0.1], 'max_depth':[3,5,10]}
xgb_grid = GridSearchCV(XGBRegressor(random_state=42, verbosity=0),
                        xgb_params, scoring='r2', cv=5, n_jobs=-1)
```

### 12. LightGBM (GridSearchCV)
```python
lgbm_params = {'n_estimators':[100,200], 'learning_rate':[0.01,0.1], 'max_depth':[3,5,10]}
lgbm_grid = GridSearchCV(LGBMRegressor(random_state=42),
                         lgbm_params, scoring='r2', cv=5, n_jobs=-1)
```

### 13. CatBoost (GridSearchCV)
```python
cat_params = {'iterations':[100,200], 'learning_rate':[0.01,0.1], 'depth':[4,6,10]}
cat_grid = GridSearchCV(CatBoostRegressor(random_state=42, verbose=0),
                        cat_params, scoring='r2', cv=5, n_jobs=-1)
```

### 14. Feature Importance (LightGBM)
```python
importances = best_lgbm.feature_importances_
importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
importance_df.sort_values('Importance', ascending=False)
# → Kariyer istatistikleri tek sezon istatistiklerini geçiyor
```

---

## 📊 Key Finding — Outlier Suppression Effect

```
# After outlier suppression, linear models OVERFIT:

Linear Regression (Suppressed):
  MAE: 233.11  |  MSE: 122555  |  R²: 0.36
  CV R²: 0.29  ← Low → Overfitting

# Ensemble modeller ise dayanıklı kaldı:
  "Best results achieved by LightGBM"
```

> **Key lesson:** Outlier suppression does not have the same effect on all models. Linear models can overfit after suppression; tree-based models are inherently more robust.

---

## 📦 Libraries Used
```python
pandas, numpy, scipy (stats), seaborn, matplotlib,
sklearn (LinearRegression, Ridge, Lasso, RandomForestRegressor,
GradientBoostingRegressor, GridSearchCV, cross_validate,
StandardScaler, train_test_split, mean_absolute_error, mean_squared_error, r2_score),
xgboost (XGBRegressor), lightgbm (LGBMRegressor), catboost (CatBoostRegressor)
```

---

## 💡 Key Takeaways
- **Outlier suppression ≠ always better** — helped ensemble models, hurt linear ones
- Career stats (CHits, CAtBat) dominate salary prediction — teams pay for track record
- LightGBM delivered the best results — best speed/accuracy tradeoff for tabular regression
- `cross_validate` is far more honest than a single train/test split for small datasets
- CatBoost: no explicit encoding needed for categoricals — handles them internally
