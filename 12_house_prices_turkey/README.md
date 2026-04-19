# 🏠 Turkey House Prices — Decision Tree & Random Forest

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![Algorithm](https://img.shields.io/badge/Algorithm-Decision%20Tree%20%7C%20Random%20Forest-orange) ![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

## 📌 Overview

End-to-end house price prediction on real Turkish real estate listing data. One of the most challenging projects in the portfolio — Turkish-formatted price strings, >66% missing value rates in some columns, high-cardinality geographic features, and a severely right-skewed price distribution. Full pipeline from raw scraping output to a tuned Random Forest with log-transformed target.

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| File | `HouseData.csv` |
| Target | `price` (TRL) |
| Task | Regression |
| Challenge | Turkish text in numeric columns, 66%+ missing, extreme outliers |

**Key features:** `district`, `price`, `GrossSquareMeters`, `NetSquareMeters`, `BuildingAge`, `NumberFloorsofBuilding`, `NumberOfRooms`, `NumberOfBathrooms`, `FloorLocation`, `InsideTheSite`, `StructureType`, `BuildStatus`

---

## 🔧 Methodology

### 1. Numeric Conversion — Turkish Format Cleaning
```python
# "12.500.000 TL" → 12500000
numeric_columns = ['price', 'GrossSquareMeters', 'NetSquareMeters', ...]
# Regex ile nokta/virgül/TL temizlendi, float'a çevrildi
```

### 2. Missing Value Strategy
```python
# %66 üstü eksik → tamamen silinir
missing_summary = df.isnull().sum() / len(df) * 100
columns_to_drop = missing_summary[missing_summary > 66].index
df = df.drop(columns=columns_to_drop)

# Gereksiz sütunlar da silindi (adres, Unnamed:0 gibi)

# Anlamlı doldurma
df['Balcony']       = df['Balcony'].fillna('Yok')          # NaN = balkon yok
df['StructureType'] = df['StructureType'].fillna('Bilinmiyor')
df['BuildStatus']   = df['BuildStatus'].fillna('Bilinmiyor')

# Seçili sütunlar: kategorik → mod, sayısal → median
for col in columns_to_fill:
    if df[col].dtype == 'object':
        df[col] = df[col].fillna(df[col].mode()[0])
    else:
        df[col] = df[col].fillna(df[col].median())
```

### 3. Outlier Analysis + Capping
```python
from scipy.stats import zscore

for col in numeric_columns:
    z_scores = np.abs(zscore(df[col]))
    # Boxplot + Z-score > 3 tespiti
    # Outliers capped (winsorization)
```

### 4. Feature Engineering
```python
df['BuildingAgeCategory'] = pd.cut(df['BuildingAge'],
    bins=[-1, 5, 20, 50, 100, 1000, float('inf')],
    labels=['0-5', '6-20', '21-50', '51-100', '100-1000', '1000+'])

df['IsInsideSite']  = df['InsideTheSite'].apply(lambda x: 1 if x == 'Evet' else 0)
df['price_per_room'] = df['price'] / df['NumberOfRooms']
df['age_density']    = df['BuildingAge'] / df['GrossSquareMeters']
df['room_density']   = df['NumberOfRooms'] / df['GrossSquareMeters']
df['price_per_sqm']  = df['price'] / df['GrossSquareMeters']
df['FloorDensity']   = df['NumberFloorsofBuilding'] / df['GrossSquareMeters']
df['AreaPerRoom']    = df['NetSquareMeters'] / df['NumberOfRooms']
```

> **Multicollinearity:** `GrossSquareMeters` and `NetSquareMeters` were highly correlated → `GrossSquareMeters` was dropped.

### 5. Correlation Analysis
```python
numeric_df = df.select_dtypes(include=['number'])
target_corr = numeric_df.corr()['price'].sort_values(ascending=False)
# Top features heatmap (ilk 10)
```

### 6. ANOVA — District Feature Significance
```python
from scipy.stats import f_oneway

anova_result = f_oneway(*[df[df['district'] == d]['price']
                           for d in df['district'].unique()])
# p-value << 0.05 → district istatistiksel olarak anlamlı ✅

# Target encoding
district_mean = df.groupby('district')['price'].mean()
df['district_encoded'] = df['district'].map(district_mean)
```

### 7. Log Transform on Target
```python
df['log_price'] = np.log1p(df['price'])
# Training on log_price; after prediction, reverse with expm1
y_pred_real = np.expm1(y_pred_logged)
```
> **Effect:** Applying log transform dramatically reduced MSE and significantly improved R².

### 8. Decision Tree
```python
from sklearn.tree import DecisionTreeRegressor

dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)
# Raw R² = 0.98 → OVERFITTING

params = {'max_depth':[3,5,10,None], 'min_samples_split':[2,5,10], 'min_samples_leaf':[1,2,4]}
grid = GridSearchCV(DecisionTreeRegressor(random_state=42), params, cv=5, scoring='r2')
grid.fit(X_train, y_train)
# R² is more balanced after GridSearch
```

### 9. Random Forest (Final Model)
```python
from sklearn.ensemble import RandomForestRegressor

rf_params = {
    'n_estimators': [100, 200],
    'max_depth':    [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf':  [1, 2]
}
rf_grid = GridSearchCV(RandomForestRegressor(random_state=42),
                       rf_params, scoring='r2', cv=5, n_jobs=-1)
rf_grid.fit(X_train_logged, y_train_logged)

cv_scores = cross_val_score(best_rf, X_logged, y_logged, cv=5, scoring='r2')
print("CV ortalama R²:", np.mean(cv_scores))

# Feature importance
importances = best_rf.feature_importances_
importance_df = pd.DataFrame({'feature': X_logged.columns, 'importance': importances})
print(importance_df.sort_values('importance', ascending=False))
```

**Feature Importance Ranking:**
1. `district_encoded` — Location dominates all other features
2. `netsquaremeters` / `price_per_sqm`
3. `district_price_density`

---

## 📦 Libraries Used
```python
pandas, numpy, re, scipy (stats.zscore, f_oneway), seaborn, matplotlib,
sklearn (DecisionTreeRegressor, RandomForestRegressor, GridSearchCV,
StandardScaler, train_test_split, mean_squared_error, r2_score, cross_val_score),
sklearn.preprocessing (LabelEncoder)
```

---

## 💡 Key Takeaways
- **Location (district)** is by far the most important feature — statistically proven by ANOVA
- Turkish numeric format cleaning (regex) is the unavoidable first step in real-world data projects
- The `66%+ missing` threshold is a practical decision, not a scientific one; can be made more aggressive if data is sufficient
- Log transform is essential for right-skewed price distributions — dramatically improves R²
- Decision Tree R²=0.98 → classic overfitting; model evaluation without cross-validation is misleading
- Random Forest training took ~40 minutes — the cost of a large grid search on real data
