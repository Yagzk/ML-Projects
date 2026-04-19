# 🏠 Turkey House Price Prediction — Decision Tree & Random Forest

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![Algorithm](https://img.shields.io/badge/Algorithm-Decision%20Tree%20%7C%20Random%20Forest-orange) ![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

## 📌 Overview

End-to-end house price prediction project using **real Turkish real estate listing data**. The dataset posed significant challenges: Turkish-formatted numbers with "TL" suffixes, high missing value rates, messy categorical columns, and highly skewed price distribution. This project covers the full pipeline — from raw messy data to a tuned Random Forest model with log-transformed target.

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| Source | Turkish real estate listings (scraped) |
| File | `HouseData.csv` |
| Target | `price` (sale price in TRL) |
| Task | Regression |
| Challenge | High missing values, Turkish text values, extreme outliers |

**Key Features:**
| Column | Description |
|--------|-------------|
| `district` | Istanbul district (ilçe) |
| `price` | Sale price (Turkish Lira) |
| `GrossSquareMeters` | Gross floor area (m²) |
| `NetSquareMeters` | Net floor area (m²) |
| `BuildingAge` | Building age in years |
| `NumberOfRooms` | Total room count |
| `NumberFloorsofBuilding` | Total floors in building |
| `FloorLocation` | Floor of the listed unit |
| `UsingStatus` | Occupancy status |
| `InsideTheSite` | Whether in a gated community |

---

## 🔧 Methodology

### 1. Data Cleaning — Turkish Numeric Formats
Columns like `price` contained values such as `"12.500.000 TL"`. These were cleaned with regex:
```python
df['price'] = df['price'].str.replace(r'[^\d]', '', regex=True).astype(float)
```

### 2. Missing Value Strategy
- Columns with **>66% missing** → **dropped entirely**
- `address`, `AdUpdateDate`, `Category` → **dropped** (irrelevant for modeling)
- `Balcony` NaN → `"Yok"` (Turkish for "None") — likely indicates no balcony
- `StructureType`, `BuildStatus` NaN → `"Bilinmiyor"` (Unknown)
- Remaining columns → mode (categorical) or median (numerical)

### 3. Outlier Analysis
- Average price: **12,870,621 TRL** with massive variance
- Z-score + boxplot detection on all numeric columns
- Applied **winsorization (clipping)** to cap extreme values

### 4. Feature Engineering

| Feature | Formula | Rationale |
|---------|---------|-----------|
| `price_per_sqm` | `price / GrossSquareMeters` | Normalizes for size — standard real estate metric |
| `BuildingAgeCategory` | `pd.cut(BuildingAge, bins)` | Captures non-linear age-price relationship |
| `IsInsideSite` | `1 if InsideTheSite == "Evet"` | Binary gated community flag |
| `price_per_room` | `price / NumberOfRooms` | Per-room value |
| `age_density` | `BuildingAge / GrossSquareMeters` | Age-adjusted density |
| `room_density` | `NumberOfRooms / GrossSquareMeters` | Space utilization |
| `FloorDensity` | `NumberFloorsofBuilding / GrossSquareMeters` | Building height ratio |

### 5. District Encoding — ANOVA + Target Encoding
```python
from scipy.stats import f_oneway

# Test if district significantly affects price
anova_result = f_oneway(*(df[df['district'] == d]['price'] for d in df['district'].unique()))
# p-value << 0.05 → district IS statistically significant
```
Used **target encoding** (district → mean price of that district) instead of one-hot, to avoid creating 30+ dummy columns:
```python
district_price_mean = df.groupby('district')['price'].mean()
df['district_encoded'] = df['district'].map(district_price_mean)
```
> 💡 *"I learned this technique from my instructor — district information is very important since it represents location."*

### 6. Log Transformation on Target
```python
df['log_price'] = np.log1p(df['price'])
```
- Prices were right-skewed (many cheap, few very expensive)
- Log transform normalized the distribution → MSE dropped dramatically
- After prediction: `np.expm1(y_pred)` to reverse the transformation

### 7. Models

**Decision Tree:**
```python
dt = DecisionTreeRegressor(random_state=42)
# Raw: R² = 0.98 → OVERFITTING
# After GridSearchCV: R² = 0.96 (more reliable)
```

**Random Forest (best model):**
```python
rf_params = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}
rf_grid = GridSearchCV(rf, rf_params, scoring='r2', cv=5, n_jobs=-1)
```
Training time: ~40 minutes. Result: *"İnanılmaz"* (Incredible) 🎉

---

## 📉 Results Summary

| Model | R² | Notes |
|-------|-----|-------|
| Decision Tree (raw) | 0.98 | Overfit |
| Decision Tree (GridSearch) | 0.96 | Better generalization |
| Decision Tree + log(price) | Higher R², lower MSE | Log transform helped significantly |
| Random Forest + log(price) | Best | Robust, low overfitting |

**Feature Importance (Random Forest top features):**
- `district_encoded` (location is king in real estate)
- `netsquaremeters` / `price_per_sqm`
- `district_price_density`

---

## 📦 Libraries Used

```python
pandas, numpy, scipy (stats.zscore, f_oneway), sklearn (DecisionTreeRegressor,
RandomForestRegressor, GridSearchCV, StandardScaler, train_test_split,
mean_squared_error, r2_score), seaborn, matplotlib
```

---

## 💡 Key Takeaways

- **Location (district)** is the single most important feature in real estate pricing — ANOVA-backed
- Log-transforming skewed targets is not optional — it's essential for tree-based models on price data
- Decision Tree R²=0.98 sounds amazing but it's overfitting — always cross-validate
- Target encoding is superior to one-hot for high-cardinality geographic features
- `GrossSquareMeters` and `NetSquareMeters` were too correlated (>0.95) — one was dropped
