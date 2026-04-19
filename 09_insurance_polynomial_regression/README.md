# 🏥 Medical Insurance Cost Prediction — Polynomial Regression & Feature Engineering

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![Algorithm](https://img.shields.io/badge/Algorithm-Polynomial%20Regression-orange) ![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

## 📌 Overview

Predicting **individual medical insurance charges** using both linear and polynomial regression. A key finding of this project is that domain-informed **interaction feature engineering** (combining smoking status with age and BMI) significantly improved model performance — to the point where polynomial transformation became unnecessary.

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| Source | Medical Cost Personal Dataset |
| File | `insurance.csv` |
| Rows | 1,338 records |
| Target | `charges` (annual insurance cost in USD) |
| Task | Regression |

**Features:**
| Column | Description |
|--------|-------------|
| `age` | Age of the primary beneficiary |
| `sex` | Gender |
| `bmi` | Body Mass Index |
| `children` | Number of covered dependents |
| `smoker` | Smoking status (yes/no) |
| `region` | Residential area (NE, NW, SE, SW) |

---

## 🔧 Methodology

### 1. Preprocessing
- Categorical encoding:
  - `sex`, `smoker`, `region` → **One-Hot Encoding** (drop_first=True)
- Numerical scaling:
  - `age`, `bmi`, `children` → **StandardScaler**

### 2. Correlation Analysis
A heatmap revealed a **very strong correlation** between `smoker_yes` and `charges`.

> 💡 *"There is a strong relationship between smoking status and charges. Also, although not as strong, there appears to be a correlation with age and BMI. Seriously, don't smoke."*

### 3. Interaction Feature Engineering

Based on the correlation insight, two **interaction features** were created:

| Feature | Formula | Rationale |
|---------|---------|-----------|
| `smoker_yes_age` | `smoker_yes × age` | Smokers' costs grow faster with age |
| `smoker_yes_bmi` | `smoker_yes × bmi` | Smokers with high BMI face compound risk |

### 4. Linear Regression (with Interaction Features)
```python
lin_model = LinearRegression()
lin_model.fit(X_train, y_train)
```
- With interaction features, R² improved dramatically
- Scatter plot showed clear clustering around the x=y line for lower values

### 5. Polynomial Regression
- Before feature engineering: optimal degree = **2** (found via GridSearchCV)
- After feature engineering: polynomial transformation was **no longer necessary**

> 📊 *"Before feature engineering, polynomial degree 2 was optimal. After feature engineering, this was no longer the case — the interaction terms captured the non-linearity."*

---

## 📉 Results

| Model | Notes |
|-------|-------|
| Linear (raw features) | Moderate R² — missing non-linear smoking effect |
| Polynomial degree=2 | Improved but more complex |
| Linear + Interaction features | Best balance — high R², interpretable |
| Polynomial + Feature Eng. | No improvement over linear + FE |

---

## 📦 Libraries Used

```python
pandas, numpy, sklearn (LinearRegression, PolynomialFeatures, StandardScaler,
GridSearchCV, train_test_split, mean_squared_error, r2_score), seaborn, matplotlib
```

---

## 💡 Key Takeaways

- **Smoking is by far the dominant predictor** of insurance costs — a single binary variable reshapes the entire model
- Interaction features (`smoker × age`, `smoker × bmi`) capture the compounding effect of smoking over time and health risk
- Smart feature engineering can replace polynomial complexity — simpler and more interpretable
- Always visualize correlations **before** choosing a modeling approach — domain insight saved significant computation here
