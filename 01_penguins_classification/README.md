# 🐧 Palmer Penguins — Species Classification

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![Algorithm](https://img.shields.io/badge/Algorithm-Classification-orange) ![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

## 📌 Overview

Multi-class classification project to predict the **species of Palmer Archipelago penguins** (Adelie, Chinstrap, Gentoo) based on physical measurements. This project covers the complete supervised learning pipeline: data loading, missing value handling, encoding, train/test splitting, and model training.

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| Source | Palmer Penguins dataset |
| Rows | 344 penguins |
| Target | `species` — Adelie / Chinstrap / Gentoo (multi-class) |
| Task | Multi-class Classification |

**Features:**
| Column | Description |
|--------|-------------|
| `bill_length_mm` | Culmen length (mm) |
| `bill_depth_mm` | Culmen depth (mm) |
| `flipper_length_mm` | Flipper length (mm) |
| `body_mass_g` | Body mass (grams) |
| `island` | Island name (Torgersen, Biscoe, Dream) |
| `sex` | Gender (male/female) |
| `year` | Study year (2007–2009) |

---

## 🔧 Methodology

### 1. Data Loading & Exploration
```python
penguins = pd.read_csv('penguins.csv')
display(penguins.head())
display(penguins.info())
display(penguins.describe())
```

### 2. Missing Value Handling
```python
# Categorical: fill with mode
penguins['sex'] = penguins['sex'].fillna(penguins['sex'].mode()[0])

# Numerical: fill with mean/median
penguins['bill_depth_mm']     = penguins['bill_depth_mm'].fillna(penguins['bill_depth_mm'].mean())
penguins['body_mass_g']       = penguins['body_mass_g'].fillna(penguins['body_mass_g'].mean())
penguins['bill_length_mm']    = penguins['bill_length_mm'].fillna(penguins['bill_length_mm'].mean())
penguins['flipper_length_mm'] = penguins['flipper_length_mm'].fillna(penguins['flipper_length_mm'].median())
```

### 3. Encoding
```python
# Binary encoding for sex
penguins['sex'] = penguins['sex'].map({'male': 0, 'female': 1})
```
`island` and `species` handled via label encoding or one-hot for downstream models.

### 4. Train/Test Split
```python
features = penguins[['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm',
                      'body_mass_g', 'island', 'sex', 'year']]
label = penguins['species']

X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.2, random_state=42)
# Train: (275, 7) | Test: (69, 7)
```

---

## 📈 Feature Insights

- **Flipper length** and **body mass** are the strongest species discriminators
- Gentoo penguins are noticeably larger than Adelie and Chinstrap
- **Island** is also highly informative — Chinstrap penguins are found almost exclusively on Dream Island
- Bill dimensions distinguish Chinstrap from Adelie well

---

## 📦 Libraries Used
```python
pandas, numpy, sklearn (train_test_split, LabelEncoder, StandardScaler)
```

---

## 💡 Key Takeaways
- Penguins is a great beginner-to-intermediate classification dataset — clean, biological, interpretable
- `sex` had missing values → mode imputation was correct (binary category, mode = 'male')
- Physical measurements are surprisingly powerful at distinguishing species — nature encodes species information into morphology
- Train/test split explanation: *"We need to test on data the model has never seen — that's the only honest way to measure how it will perform in the real world"*
