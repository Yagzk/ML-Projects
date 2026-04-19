# 🎓 University Admission — Advanced Logistic Regression (odev12 + odev13)

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![Algorithm](https://img.shields.io/badge/Algorithm-Logistic%20Regression%20%7C%20SMOTE%20%7C%20GridSearch-orange) ![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

## 📌 Overview

Two-notebook progression using the same admission dataset with increasing complexity. **odev12** builds the baseline VIF + Logistic Regression pipeline with SGD and L1/L2 comparison. **odev13** raises the admission threshold to 0.80, adds outlier removal (IQR), applies SMOTE for class imbalance, and compares GridSearchCV vs RandomizedSearchCV for hyperparameter tuning.

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| File | `Admission_Predict.csv` |
| Rows | 500 applicants |
| Task | Binary Classification |

| Notebook | Threshold | Key Addition |
|----------|-----------|--------------|
| odev12 | ≥ 0.75 | VIF, SGD, L1/L2 regularization |
| odev13 | ≥ 0.80 | Outlier removal, SMOTE, GridSearch vs RandomizedSearch |

---

## 📂 Notebooks

### `1__logistic_parameter.ipynb` (odev12) — Baseline + Regularization Comparison

Same as `10_university_admission_logistic` — VIF analysis, baseline LogisticRegression, SGD, and L1/L2 regularization on the ≥0.75 threshold.

See [10_university_admission_logistic](../10_university_admission_logistic/) for full details.

---

### `2__admission.ipynb` (odev13) — Advanced Pipeline

**Target:** `Chance of Admit ≥ 0.80` → more selective, creates class imbalance

#### 1. Outlier Detection — Boxplot + Z-Score
```python
# Tüm sayısal değişkenler için yatay boxplot
sns.boxplot(data=data[numeric_cols], orient="h", palette="Set2")

# Z-score kontrolü
z_scores = np.abs(stats.zscore(data["CGPA"]))
outliers_z = data[z_scores > 3]
print("Z-Score outlier count:", outliers_z.shape[0])
```

#### 2. IQR Outlier Removal
```python
def remove_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return data[(data[column] >= lower) & (data[column] <= upper)]

for col in X:
    data = remove_outliers_iqr(data, col)
```

#### 3. SMOTE — Class Imbalance Handling
```python
from collections import Counter
from imblearn.over_sampling import SMOTE

print("Before SMOTE:", Counter(y_train))

smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

print("After SMOTE:", Counter(y_train))
```
> ⚠️ SMOTE was applied **only to the training set** — the test set remains untouched.
> **Order matters:** SMOTE → then StandardScaler

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
```

#### 4. GridSearchCV vs RandomizedSearchCV Karşılaştırması
```python
# GridSearch
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10]}
log_reg = LogisticRegression(multi_class='multinomial', solver='lbfgs',
                              max_iter=500, class_weight='balanced')
grid_search = GridSearchCV(log_reg, param_grid, cv=5)

# RandomizedSearch
random_search = RandomizedSearchCV(log_reg, param_distributions, n_iter=50, cv=5)
```

#### 5. Tüm Modellerin Karşılaştırması
```python
accuracies = [base_accuracy, grid_accuracy, random_accuracy]
models = ['Default', 'GridSearch', 'RandomizedSearch']
plt.bar(models, accuracies)
```

**Metric comparison:**
```python
def get_classification_metrics(model, X_test, y_test):
    metrics = {
        "Accuracy":  accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average='macro'),
        "Recall":    recall_score(y_test, y_pred, average='macro'),
        "F1":        f1_score(y_test, y_pred, average='macro'),
        "ROC-AUC":   roc_auc_score(y_test, y_pred)
    }
```

> **Finding:** RandomizedSearch outperformed GridSearch. RandomizedSearch with the `liblinear` solver (ideal for small datasets + L1) produced the most consistent results.

#### 6. Confusion Matrix Karşılaştırması (3 Model)
```python
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
# Base | GridSearch | RandomizedSearch confusion matrices yan yana
```

---

## 🆚 odev12 vs odev13

| Aspect | odev12 | odev13 |
|--------|--------|--------|
| Admission threshold | ≥ 0.75 | ≥ 0.80 (more selective) |
| Outlier analysis | None | Boxplot + Z-score + IQR removal |
| Class imbalance | Not handled | SMOTE applied |
| Hyperparameter tuning | Default | GridSearch + RandomizedSearch |
| Evaluation | Accuracy, F1 | Accuracy, Precision, Recall, F1, ROC-AUC |

---

## 📦 Libraries Used
```python
pandas, numpy, scipy (stats), seaborn, matplotlib, collections (Counter),
sklearn (LogisticRegression, SGDClassifier, StandardScaler, GridSearchCV,
RandomizedSearchCV, train_test_split, accuracy_score, classification_report,
confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score),
imblearn (SMOTE), statsmodels (VIF)
```

---

## 💡 Key Takeaways
- Raising the threshold from 0.75→0.80 increases class imbalance → SMOTE becomes necessary
- **Order of SMOTE matters:** Split → SMOTE → Scale (reverse order causes data leakage)
- RandomizedSearch outperformed GridSearch — preferred for large parameter spaces
- `average='macro'` → all classes weighted equally, correct choice for imbalanced data
- IQR > Z-score: Z-score assumes normality on small datasets; IQR is more robust
