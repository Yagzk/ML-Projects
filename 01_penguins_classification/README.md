# 🐧 Palmer Penguins — Data Preprocessing & Classification Preparation

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

## 📌 Overview

First project of the portfolio. The Palmer Penguins dataset is used to practice the **complete data preparation pipeline** for multi-class classification: loading, exploring, handling missing values, encoding, and splitting into train/test sets. This project establishes the preprocessing foundation that all later classification projects build upon.

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| Source | Palmer Penguins (CSV) |
| Rows | 344 penguins |
| Target | `species` — Adelie / Chinstrap / Gentoo |
| Task | Multi-class Classification Preparation |

**Features:**
| Column | Description |
|--------|-------------|
| `bill_length_mm` | Bill length (mm) |
| `bill_depth_mm` | Bill depth (mm) |
| `flipper_length_mm` | Flipper length (mm) |
| `body_mass_g` | Body mass (grams) |
| `island` | Island (Torgersen / Biscoe / Dream) |
| `sex` | Gender (male / female) |
| `year` | Study year (2007–2009) |

---

## 🔧 Methodology

### 1. Exploration
```python
penguins.info()      # 344 rows, data types, NaN counts
penguins.describe()  # Statistical summary of numerical columns
penguins.isnull().sum()
```

### 2. Missing Value Handling
```python
# Categorical → mode
penguins['sex'] = penguins['sex'].fillna(penguins['sex'].mode()[0])

# Numerical → mean / median
penguins['bill_depth_mm']     = penguins['bill_depth_mm'].fillna(penguins['bill_depth_mm'].mean())
penguins['body_mass_g']       = penguins['body_mass_g'].fillna(penguins['body_mass_g'].mean())
penguins['bill_length_mm']    = penguins['bill_length_mm'].fillna(penguins['bill_length_mm'].mean())
penguins['flipper_length_mm'] = penguins['flipper_length_mm'].fillna(penguins['flipper_length_mm'].median())
```

### 3. Encoding
```python
penguins['sex'] = penguins['sex'].map({'male': 0, 'female': 1})
```

### 4. Train/Test Split
```python
from sklearn.model_selection import train_test_split

features = penguins[['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm',
                      'body_mass_g', 'island', 'sex', 'year']]
label = penguins['species']

X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.2, random_state=42)
# X_train: (275, 7) | X_test: (69, 7)
```

---

## 📦 Libraries Used
```python
pandas, numpy, sklearn (train_test_split)
```

---

## 💡 Key Takeaways
- `sex` had missing values → mode imputation is correct for binary categoricals
- `flipper_length_mm` used median (more robust than mean for skewed distributions)
- Train/test split rationale: *"The model must be tested on data it has never seen — that's the only honest way to measure real-world performance"*
- Flipper length and body mass are the strongest species discriminators — Gentoo penguins are noticeably larger
- Island is highly informative — Chinstrap penguins are found almost exclusively on Dream Island
