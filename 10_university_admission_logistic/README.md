# 🎓 University Admission Prediction — Logistic Regression

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![Algorithm](https://img.shields.io/badge/Algorithm-Logistic%20Regression-orange) ![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

## 📌 Overview

Binary classification model to predict whether a graduate school applicant has a **≥75% chance of admission** based on academic and research profile. The project includes thorough VIF-based multicollinearity analysis before fitting the logistic regression model.

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| Source | Graduate Admissions Dataset |
| File | `Admission_Predict.csv` |
| Rows | 500 applicants |
| Target | Binary: `Chance of Admit ≥ 0.75` → 1, else 0 |
| Task | Binary Classification |

**Features:**
| Column | Description | Range |
|--------|-------------|-------|
| `GRE Score` | Graduate Record Exam score | 0–340 |
| `TOEFL Score` | English proficiency score | 0–120 |
| `University Rating` | University prestige rating | 1–5 |
| `SOP` | Statement of Purpose quality | 1–5 |
| `LOR` | Letter of Recommendation strength | 1–5 |
| `CGPA` | Undergraduate GPA | 1–10 |
| `Research` | Research experience (0/1) | Binary |

---

## 🔧 Methodology

### 1. Target Variable Creation
The original `Chance of Admit` is a continuous probability. It was binarized:
```python
data["Chance of Admit Binary"] = (data["Chance of Admit "] >= 0.75).astype(int)
```

### 2. Correlation Analysis
A heatmap showed **high correlations** among GRE Score, TOEFL Score, and CGPA — raising multicollinearity concerns.

### 3. VIF Analysis (Variance Inflation Factor)
```python
from statsmodels.stats.outliers_influence import variance_inflation_factor

V = add_constant(data[["GRE Score", "TOEFL Score", "University Rating",
                        "SOP", "LOR ", "CGPA", "Research"]])
vif_data["VIF"] = [variance_inflation_factor(V.values, i) for i in range(V.shape[1])]
```

**Decision:** All VIF values remained below 5 — no feature needed to be removed except `LOR`, which had:
- Low correlation with the binary target
- Moderate overlap with `SOP`

→ **`LOR` was dropped** as the weakest contributor.

### 4. Logistic Regression
```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train_scaled, y_train)
```

### 5. Evaluation
- Confusion Matrix
- Accuracy, Precision, Recall, F1 Score
- Classification Report

---

## 📦 Libraries Used

```python
pandas, numpy, sklearn (LogisticRegression, StandardScaler, train_test_split,
confusion_matrix, classification_report), statsmodels (VIF), seaborn, matplotlib
```

---

## 💡 Key Takeaways

- **CGPA** emerged as the strongest single predictor of admission probability
- GRE and TOEFL scores are strongly correlated with each other but both remained meaningful enough to keep (VIF < 5)
- VIF analysis is crucial for logistic regression — high multicollinearity inflates standard errors and destabilizes coefficient estimates
- Removing `LOR` (Letter of Recommendation) was a data-driven decision, not an assumption
- Binary classification at a 0.75 probability threshold worked well — the natural gap in `Chance of Admit` distribution aligned with this cutoff
