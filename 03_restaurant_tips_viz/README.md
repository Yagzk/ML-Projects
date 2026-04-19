# 🍽️ Restaurant Tips — Exploratory Data Analysis & Visualization

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

## 📌 Overview

A deep-dive **EDA and visualization** project using the classic Seaborn `tips` dataset. The goal is to uncover patterns in tipping behavior based on meal time, party size, smoking status, day of the week, and total bill amount — using a range of statistical visualizations.

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| Source | Seaborn built-in (`sns.load_dataset("tips")`) |
| Rows | 244 restaurant transactions |
| Features | `total_bill`, `tip`, `sex`, `smoker`, `day`, `time`, `size` |

**Variable descriptions:**
- `total_bill`: Total bill amount (USD)
- `tip`: Tip amount left by the customer
- `sex`: Customer gender (Male/Female)
- `smoker`: Whether the customer smokes (Yes/No)
- `day`: Day of the week (Thu, Fri, Sat, Sun)
- `time`: Meal time (Lunch/Dinner)
- `size`: Table size (number of people)

---

## 📈 Visualizations & Findings

### 1. Scatter Plot — Total Bill vs Tip (by Smoker Status)
- **Finding:** As total bill increases, tip amount generally increases as well (positive correlation)
- Smoking status does **not** have a direct effect on tip amount

### 2. Box Plot — Total Bill Distribution by Day
- **Finding:** Saturday and Sunday have the highest median bills
- Thursday shows the most outliers
- Weekend dinners drive up the average spend

### 3. Bar Chart — Average Tip Rate by Party Size
```
tip_rate = (tip / total_bill) × 100
```
- **Finding:** Tip rate **decreases** as party size grows
- Exception: 5-person tables tip slightly more than 4 and 6-person tables

### 4. Histogram + KDE — Lunch vs Dinner Tip Distribution
- **Finding:** Dinner tips are higher and more spread out; lunch tips are narrower and more predictable

### 5. Heatmap — Correlation Matrix
- **Strong correlation:** `total_bill` ↔ `tip` (r ≈ 0.68)
- **Moderate correlation:** `total_bill` ↔ `size` (r ≈ 0.60)
- Other correlations remain below 0.5

---

## 🔧 Techniques Used

```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.scatterplot()    # Tip vs bill (colored by smoker)
sns.boxplot()        # Bill by day
sns.barplot()        # Tip rate by size
sns.histplot()       # Lunch vs Dinner distribution
sns.heatmap()        # Correlation matrix
```

---

## 💡 Key Takeaways

- Total bill is the strongest predictor of tip amount
- Larger parties tend to leave proportionally smaller tips — group dynamics matter
- Dinner sessions generate more revenue and higher tips than lunch sessions
- Smoking status is a surprisingly weak predictor of tipping behavior
