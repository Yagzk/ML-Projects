# 📐 Machine Learning Evaluation Metrics — Theory & Practice

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![Type](https://img.shields.io/badge/Type-Theory%20%2B%20Examples-purple) ![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

## 📌 Overview

A comprehensive theoretical reference notebook covering all key **evaluation metrics** used in machine learning — for both classification and regression problems. Each metric is explained with real-world analogies, formulas, and practical decision guidelines.

---

## 📋 Classification Metrics

### ✅ Accuracy
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```
**When to use:** Balanced datasets where all errors are equally costly.  
**When NOT to use:** Imbalanced datasets — a model that always predicts the majority class can achieve high accuracy while being completely useless.

> **Example:** A factory quality control model with 50% defective rate. Accuracy is meaningful here because classes are balanced.

---

### 🎯 Precision
```
Precision = TP / (TP + FP)
```
**When to use:** False Positives are expensive. You want to minimize cases where the model incorrectly flags something as positive.  
> **Example:** Email spam filter — you don't want to classify a legitimate important email as spam (FP).

---

### 🔍 Recall (Sensitivity)
```
Recall = TP / (TP + FN)
```
**When to use:** False Negatives are dangerous. Missing a positive case is costly.  
> **Example:** Cancer screening — falsely clearing a sick patient (FN) has life-threatening consequences.

---

### ⚖️ F1 Score
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```
**When to use:** Imbalanced datasets where you need a balance between Precision and Recall.  
> **Example:** Machine fault detection — minimize false alarms (Precision) while still catching real faults (Recall).

---

### 🧮 Confusion Matrix

|  | Predicted Positive | Predicted Negative |
|--|---|---|
| **Actual Positive** | TP ✅ | FN ❌ (Type II Error) |
| **Actual Negative** | FP ❌ (Type I Error) | TN ✅ |

---

### 🎓 Real-World Scenario: Why 99% Accuracy Can Be Misleading

```
Dataset: 1000 cancer screenings
TP = 960  (correctly identified cancer)
TN = 30   (correctly cleared healthy patients)
FP = 0
FN = 10   (missed cancer patients → sent home!)

Accuracy = 990/1000 = 99% → Looks great!
Recall   = 960/970  = 98.97% → Also good
BUT: 10 sick people were cleared and sent home — potentially fatal.
```

**Lesson:** Always check Recall for medical/safety applications, not just Accuracy.

---

## 📈 Regression Metrics

### MAE (Mean Absolute Error)
```
MAE = (1/n) × Σ|y_actual - y_predicted|
```
- Treats all errors equally (no penalty for large errors)
- Easy to interpret in original units
- **Best for:** Delivery time estimation, when small and large errors are equally acceptable

---

### MSE (Mean Squared Error)
```
MSE = (1/n) × Σ(y_actual - y_predicted)²
```
- Penalizes large errors more heavily due to squaring
- Harder to interpret (units are squared)

---

### RMSE (Root Mean Squared Error)
```
RMSE = √MSE
```
- Same unit as the target variable → easier to interpret than MSE
- Still penalizes large errors strongly
- **Best for:** Drug dosage prediction, house price estimation, any domain where large errors are dangerous

---

### R² (Coefficient of Determination)
```
R² = 1 - (SS_residual / SS_total)
```
- R² = 1.0 → perfect model; R² = 0 → model learned nothing
- **Example:** A bank credit score model with R² = 0.92 explains 92% of the variance in credit scores

---

## 🤔 Decision Guide

| Scenario | Recommended Metric |
|----------|-------------------|
| Balanced classification | Accuracy |
| Spam detection (minimize false positives) | Precision |
| Cancer screening (don't miss sick patients) | Recall |
| Imbalanced dataset | F1 Score |
| House price prediction | RMSE |
| Delivery time estimation | MAE |
| General model quality | R² |
| Large errors are catastrophic | RMSE |

---

## 💡 Key Takeaways

1. **No single metric tells the whole story** — always evaluate multiple metrics
2. **Domain knowledge drives metric choice** — a 10% error in drug dosage is very different from a 10% error in delivery time
3. **Imbalanced data breaks Accuracy** — use F1, Precision, Recall, or AUC-ROC instead
4. **RMSE > MAE** when outlier penalties matter; **MAE > RMSE** when all errors are equal
