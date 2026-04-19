# 🎓 University Admission — Logistic Regression + Regularization

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![Algorithm](https://img.shields.io/badge/Algorithm-Logistic%20Regression%20%7C%20SGD-orange) ![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

## 📌 Overview

Binary classification predicting graduate school admission (≥75% chance). Beyond baseline Logistic Regression, this project also explores **Stochastic Gradient Descent (SGD)**, and compares **L1 (Lasso) vs L2 (Ridge) regularization** applied to logistic regression — an important step toward understanding how regularization affects classification, not just regression.

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| File | `Admission_Predict.csv` |
| Rows | 500 applicants |
| Target | `Chance of Admit ≥ 0.75` → 1, else 0 |
| Task | Binary Classification |

**Features used:** `GRE Score`, `TOEFL Score`, `University Rating`, `SOP`, `CGPA`, `Research`
*(LOR was removed — low correlation with binary target + overlap with SOP)*

---

## 🔧 Methodology

### 1. Target Variable
```python
data["Chance of Admit Binary"] = (data["Chance of Admit "] >= 0.75).astype(int)
```

### 2. Correlation + VIF Analysis
```python
from statsmodels.stats.outliers_influence import variance_inflation_factor

V = add_constant(data[["GRE Score", "TOEFL Score", "University Rating",
                        "SOP", "LOR ", "CGPA", "Research"]])
vif_data["VIF"] = [variance_inflation_factor(V.values, i) for i in range(V.shape[1])]
```
> All VIF values remained below 5 → GRE and TOEFL were kept. LOR was removed (low correlation with target + overlap with SOP).

### 3. Train/Test Split + Scaling
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
```

### 4. Baseline Logistic Regression
```python
model = LogisticRegression(max_iter=1000, tol=1e-4, solver='lbfgs', random_state=42)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
```
> `tol=1e-4` → consecutive log-loss difference below this threshold triggers early stopping.

**Predict probabilities table:**
```python
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
results_df = pd.DataFrame({
    "Actual Value": y_test.values.flatten(),
    "Predicted Probability": y_pred_proba.round(4),
    "Tahmin": y_pred
})
```

### 5. SGD Classifier (Stochastic Gradient Descent)
```python
from sklearn.linear_model import SGDClassifier

sgd_model = SGDClassifier(loss="log_loss", max_iter=1000, tol=1e-4,
                           random_state=42, verbose=1)
sgd_model.fit(X_train_scaled, y_train.values.ravel())
```
> **Finding:** Model stopped early at epoch 26 — loss improved by less than 0.001 over the previous 5 epochs. The ideal stopping point was epoch 54.

### 6. L2 (Ridge) Regularization
```python
model_l2 = LogisticRegression(penalty='l2', solver='lbfgs', max_iter=1000)
model_l2.fit(X_train_scaled, y_train)
y_pred_l2 = model_l2.predict(X_test_scaled)

print(f"L2 Accuracy: {accuracy_score(y_test, y_pred_l2) * 100:.2f}%")
pd.DataFrame(model_l2.coef_[0], index=X.columns, columns=["L2 Weight"])
```

### 7. L1 (Lasso) Regularization
```python
model_l1 = LogisticRegression(penalty='l1', solver='liblinear', max_iter=1000)
model_l1.fit(X_train_scaled, y_train)
y_pred_l1 = model_l1.predict(X_test_scaled)

print(f"L1 Accuracy: {accuracy_score(y_test, y_pred_l1) * 100:.2f}%")
pd.DataFrame(model_l1.coef_[0], index=X.columns, columns=["L1 Weight"])
```
> **Finding:** L1 and L2 yielded identical accuracy scores. Reason: no overfitting, clean and small dataset. However, **the weights differed** — L1 pushed some feature coefficients toward zero.

---

## 📦 Libraries Used
```python
pandas, numpy, seaborn, matplotlib,
sklearn (LogisticRegression, SGDClassifier, StandardScaler, train_test_split,
accuracy_score, classification_report, confusion_matrix),
statsmodels (variance_inflation_factor)
```

---

## 💡 Key Takeaways
- `solver='lbfgs'` — designed for large datasets, stable on small ones too
- `solver='liblinear'` — ideal for small datasets with L1 regularization
- SGD is much faster than batch gradient descent on large datasets; early stopping prevents overfitting
- L1 and L2 produced identical accuracy → regularization effect is limited on clean, balanced datasets
- LOR was removed despite VIF < 5 — VIF alone is not enough; correlation with the target must also be checked
