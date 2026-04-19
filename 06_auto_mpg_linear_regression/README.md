# 🚗 Auto MPG — Linear Regression with VIF & Feature Engineering

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![Algorithm](https://img.shields.io/badge/Algorithm-Linear%20Regression-orange) ![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

## 📌 Overview

Predicting a vehicle's **fuel efficiency (MPG — miles per gallon)** using the UCI Auto MPG dataset. This project focuses on diagnosing and resolving **multicollinearity** using VIF (Variance Inflation Factor) analysis, followed by feature engineering to build a cleaner, more reliable linear regression model.

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| Source | UCI Machine Learning Repository (Auto MPG, id=9) |
| Rows | 392 vehicles |
| Target | `mpg` (miles per gallon — continuous) |
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
- `horsepower` had 6 missing values → filled with **median**
- `origin` is nominal (countries) → applied **One-Hot Encoding** to avoid false ordinal relationships

### 2. Train/Test Split Before Scaling
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# ✅ Scaling AFTER split to prevent data leakage
scaler.fit_transform(X_train)
scaler.transform(X_test)
```

### 3. Baseline Linear Regression
- Trained on continuous features: `displacement`, `horsepower`, `weight`, `acceleration`
- Evaluated with MAE, MSE, R²
- Visualized predictions vs actual values using scatter plot

### 4. Multicollinearity Detection — VIF Analysis
```
High VIF detected (VIF > 10):
- displacement  → very high
- cylinders     → very high
- weight        → very high
- horsepower    → high
```
Conclusion: **Strong multicollinearity** among engine-related features reduces model reliability.

### 5. Feature Engineering to Reduce VIF

| New Feature | Formula | Rationale |
|-------------|---------|-----------|
| `displacement_per_cyl` | `displacement / cylinders` | Combines two correlated features into one |
| `acceleration_per_weight` | `acceleration / weight` | Power-to-weight ratio proxy |

**Result:** Dropped `horsepower`, `weight`, `displacement`, `acceleration`, `model_year`, `cylinders` → significantly reduced VIF values.

> **Note:** `model_year` was also removed as it consistently increased VIF regardless of transformations.

---

## 📉 Results

| Model | Notes |
|-------|-------|
| Baseline (raw features) | High multicollinearity, inflated coefficients |
| After VIF + Feature Eng. | Lower VIF, more stable and interpretable model |

---

## 📦 Libraries Used

```python
pandas, numpy, sklearn (LinearRegression, StandardScaler, train_test_split),
statsmodels (variance_inflation_factor), matplotlib, seaborn
```

---

## 💡 Key Takeaways

- **Multicollinearity** doesn't hurt predictions much but destroys coefficient interpretability
- VIF > 10 signals a serious problem — either remove the feature or engineer a combined one
- One-Hot Encoding for `origin` was crucial — treating countries as 1/2/3 would imply Japan > Europe > USA, which is meaningless
- Always split data before scaling to avoid information leakage from the test set
