# 📺 Advertising — Linear Regression (odev8)

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![Algorithm](https://img.shields.io/badge/Algorithm-Linear%20Regression-orange) ![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

## 📌 Overview

Predicting **product sales** based on advertising spend across three channels — TV, Radio, and Newspaper — using Multiple Linear Regression. The trained model is serialized with joblib for deployment.

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| File | `Advertising.csv` |
| Rows | 200 records |
| Target | `Sales` (product sales units) |
| Task | Regression |

**Features:**
| Column | Description |
|--------|-------------|
| `TV` | Advertising budget spent on TV (thousands USD) |
| `Radio` | Advertising budget spent on Radio (thousands USD) |
| `Newspaper` | Advertising budget spent on Newspaper (thousands USD) |

---

## 🔧 Methodology

### 1. Exploratory Visualization
Scatter plots for each channel vs Sales:
```python
for column in ['TV', 'Radio', 'Newspaper']:
    plt.scatter(data[column], data['Sales'])
```
**Finding:** TV has the strongest linear relationship with Sales. Newspaper shows the weakest correlation.

### 2. Model Training
```python
from sklearn.linear_model import LinearRegression

X = data[['TV', 'Radio', 'Newspaper']]
y = data['Sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
```

### 3. Model Coefficients
| Feature | Coefficient | Interpretation |
|---------|-------------|----------------|
| `TV` | 0.0447 | Every 1 unit increase in TV spend → 0.045 unit increase in Sales |
| `Radio` | 0.1892 | Every 1 unit increase in Radio spend → 0.189 unit increase in Sales |
| `Newspaper` | ~0.00 | Virtually no effect on Sales |

**Insight:** Radio has ~4x more impact per unit than TV. Newspaper barely moves the needle.

### 4. Evaluation
```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2  = r2_score(y_test, y_pred)
```

### 5. Model Serialization
```python
import joblib
joblib.dump(model, 'advertising_model.joblib')
```

---

## 📦 Libraries Used
```python
pandas, numpy, matplotlib, sklearn (LinearRegression, train_test_split,
mean_squared_error, mean_absolute_error, r2_score), joblib
```

---

## 💡 Key Takeaways
- TV has the highest raw budget allocation but Radio delivers higher sales impact per dollar spent
- Newspaper spending shows almost zero effect on sales — a clear budget reallocation insight
- `joblib` serialization makes the model reusable without retraining — essential for deployment
- This project follows Auto MPG (odev7) — same algorithm, different domain, reinforcing linear regression fundamentals
