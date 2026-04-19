# 🎓 University Admission — Advanced Logistic Regression (1) logistic parameter + 2) admission)

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![Algorithm](https://img.shields.io/badge/Algorithm-Logistic%20Regression-orange) ![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

## 📌 Overview

Two-notebook progression on graduate school admission prediction — both using the same dataset but tackling increasingly complex challenges. **1) logistic parameter** focuses on VIF-based feature selection and baseline logistic regression. **2) admission** extends this with outlier removal (IQR), a stricter admission threshold, and SMOTE to handle class imbalance.

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| Source | Graduate Admissions Dataset |
| File | `Admission_Predict.csv` |
| Rows | 500 applicants |
| Task | Binary Classification |

**Features:**
| Column | Description | Range |
|--------|-------------|-------|
| `GRE Score` | Graduate Record Exam score | 0–340 |
| `TOEFL Score` | English proficiency score | 0–120 |
| `University Rating` | University prestige | 1–5 |
| `SOP` | Statement of Purpose quality | 1–5 |
| `LOR` | Letter of Recommendation strength | 1–5 |
| `CGPA` | Undergraduate GPA | 1–10 |
| `Research` | Research experience | 0/1 |

---

## 📂 Notebooks

### v1 — `1) logistic parameter` · Baseline + VIF Feature Selection

**Target:** `Chance of Admit ≥ 0.75` → binary (0/1)

**Pipeline:**
1. EDA — korelasyon heatmap, tüm değerler numerik, eksik değer yok
2. **VIF Analizi** — multikollinearite kontrolü
```python
from statsmodels.stats.outliers_influence import variance_inflation_factor
V = add_constant(data[["GRE Score", "TOEFL Score", "University Rating",
                        "SOP", "LOR ", "CGPA", "Research"]])
vif_data["VIF"] = [variance_inflation_factor(V.values, i) for i in range(V.shape[1])]
```
**Karar:** Tüm VIF değerleri 5'in altında → GRE ve TOEFL çıkarılmadı. Ancak **LOR** çıkarıldı — hem binary ile korelasyonu düşük, hem de SOP ile örtüşüyor.

3. Logistic Regression eğitimi, confusion matrix, classification report

---

### v2 — `odev_13.ipynb` · Outlier Removal + SMOTE

**Target:** `Chance of Admit ≥ 0.80` → daha seçici eşik, daha dengesiz sınıf

**v1'e göre eklemeler:**

#### Aykırı Değer Tespiti — Boxplot + Z-Score
```python
# Tüm sayısal değişkenler için yatay boxplot
sns.boxplot(data=data[numeric_cols], orient="h", palette="Set2")

# Z-score ile kontrol
z_scores = np.abs(stats.zscore(data["CGPA"]))
outliers_z = data[z_scores > 3]
```

#### IQR ile Aykırı Değer Temizleme
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

#### Sınıf Dengesizliği Kontrolü + SMOTE
```python
from collections import Counter
print("SMOTE öncesi:", Counter(y_train))

from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
print("SMOTE sonrası:", Counter(y_train_resampled))
```
> ⚠️ SMOTE sadece training set'e uygulandı — test seti orijinal haliyle kaldı.

---

## 🆚 1) logistic parameter vs 2) admission Karşılaştırması

| Aspect | 1) logistic parameter | 2) admission |
|--------|--------|--------|
| Kabul eşiği | ≥ 0.75 | ≥ 0.80 (daha seçici) |
| Aykırı değer işleme | Yok | Z-score + IQR temizleme |
| Sınıf dengesizliği | Ele alınmadı | SMOTE uygulandı |
| VIF analizi | ✅ | ✅ |
| Zorluk seviyesi | Orta | İleri |

---

## 📦 Libraries Used
```python
pandas, numpy, scipy (stats.zscore), sklearn (LogisticRegression, StandardScaler,
train_test_split, classification_report, confusion_matrix),
imblearn (SMOTE), statsmodels (VIF), seaborn, matplotlib, collections (Counter)
```

---

## 💡 Key Takeaways
- Eşiği 0.75'ten 0.80'e çıkarmak sınıf dengesizliğini artırır — SMOTE zorunlu hale gelir
- **IQR > Z-score** — küçük veri setlerinde Z-score normallik varsayar, IQR daha sağlamdır
- SMOTE yalnızca training set'e uygulanmalı — test seti asla augment edilmemeli
- LOR'un çıkarılması veri destekli bir karar — sezgiye değil VIF + korelasyon analizine dayalı
