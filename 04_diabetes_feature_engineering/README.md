# 🩺 Diabetes Prediction — Feature Engineering + Model Training

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![Algorithm](https://img.shields.io/badge/Algorithm-Random%20Forest%20%7C%20Logistic%20Regression-orange) ![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

## 📌 Overview

End-to-end diabetes prediction project on the Pima Indians Diabetes dataset. The focus is on **feature engineering** — creating new meaningful variables — followed by scaling, correlation analysis, and model training with both Random Forest and Logistic Regression.

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| Source | Pima Indians Diabetes Database |
| File | `diabetes.csv` |
| Rows | 768 patients |
| Target | `Outcome` (1 = Diabetic, 0 = Non-Diabetic) |
| Task | Binary Classification |

**Features:**
| Column | Description |
|--------|-------------|
| `Pregnancies` | Number of pregnancies |
| `Glucose` | Plasma glucose concentration |
| `BloodPressure` | Diastolic blood pressure (mm Hg) |
| `SkinThickness` | Triceps skin fold thickness (mm) |
| `Insulin` | 2-Hour serum insulin (mu U/ml) |
| `BMI` | Body Mass Index |
| `DiabetesPedigreeFunction` | Diabetes family history score |
| `Age` | Age in years |

---

## 🔧 Methodology

### 1. Zero Value Treatment
Biologically impossible zeros in medical columns replaced with median:
```python
missing_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
(df[missing_cols] == 0).sum()  # Check zero counts

for col in missing_cols:
    df[col] = df[col].replace(0, df[col].median())
```

### 2. Feature Engineering

| Feature | Formula | Rationale |
|---------|---------|-----------|
| `AgeCategory` | `pd.cut(Age, bins=[0,12,18,35,55,100])` | Non-linear age effects: Baby/Child/Young/Middle-Aged/Elderly |
| `Pregnancy_Age_Ratio` | `Pregnancies / Age` | Relative pregnancy burden by age |
| `HealthRiskScore` | `BMI × Glucose` | Combined metabolic risk indicator |

```python
df['AgeCategory'] = pd.cut(df['Age'],
                            bins=[0, 12, 18, 35, 55, 100],
                            labels=["Bebek", "Çocuk", "Genç", "Orta Yaş", "Yaşlı"])

df['Pregnancy_Age_Ratio'] = df['Pregnancies'] / df['Age']
df['HealthRiskScore']     = df['BMI'] * df['Glucose']
```

### 3. Standardization
```python
from sklearn.preprocessing import StandardScaler

numerical_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                     'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age',
                     'Pregnancy_Age_Ratio', 'HealthRiskScore']
scaler = StandardScaler()
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
```

### 4. Correlation Analysis
```python
correlation_matrix = numeric_df.corr()
target_corr = correlation_matrix['Outcome'].sort_values(ascending=False)
```
**Result:** `HealthRiskScore` showed the highest correlation with `Outcome`.

### 5. Encoding
```python
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
df['AgeCategory'] = label_encoder.fit_transform(df['AgeCategory'])
```

### 6. Train/Test Split
```python
Y = df['Outcome']
X = df.drop('Outcome', axis=1)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20,
                                                      random_state=0, stratify=Y)
```
> `stratify=Y` used to preserve the class distribution in both splits.

### 7. Model Training

#### Random Forest
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, Y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(Y_test, y_pred)
print(f"Random Forest Accuracy: {accuracy:.4f}")
```

#### Logistic Regression
```python
from sklearn.linear_model import LogisticRegression

model_lr = LogisticRegression()
model_lr.fit(X_train, Y_train)
accuracy_lr = accuracy_score(Y_test, y_pred)
print(f"Logistic Regression Accuracy: {round(accuracy_lr, 2) * 100}%")
```

---

## 📦 Libraries Used
```python
pandas, numpy, seaborn, matplotlib,
sklearn (RandomForestClassifier, LogisticRegression, StandardScaler,
LabelEncoder, train_test_split, accuracy_score)
```

---

## 💡 Key Takeaways
- Zero values in medical data are missing values in disguise — domain knowledge is essential
- `HealthRiskScore = BMI × Glucose` outperforms either feature alone in correlation with the target
- `stratify=Y` in train/test split preserves class ratio — critical for imbalanced medical datasets
- Random Forest and Logistic Regression both trained and evaluated — comparison shows the gap between ensemble and linear approaches
- Feature engineering done **before** standardization so engineered features also get scaled correctly
