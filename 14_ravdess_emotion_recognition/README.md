# 🎙️ RAVDESS — Speech Emotion Recognition

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![Algorithm](https://img.shields.io/badge/Algorithm-SVM%20%7C%20MFCC-orange) ![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

## 📌 Overview

Audio-based emotion classification using the RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song) dataset. Speech features are extracted using **MFCC (Mel-Frequency Cepstral Coefficients)** via librosa, and a **Support Vector Machine (SVM)** is trained inside a sklearn Pipeline with GridSearchCV optimization. The final model is serialized with joblib and can predict emotions from new `.wav` files.

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| Source | RAVDESS Dataset |
| Format | `.wav` audio files |
| Emotions | Neutral, Calm, Happy, Sad, Angry, Fearful, Disgust, Surprised (8 classes) |
| Task | Multi-class Audio Classification |

---

## 🔧 Methodology

### 1. Feature Extraction — MFCC
```python
import librosa

def extract_features(file_path):
    audio, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    return np.mean(mfcc.T, axis=0)  # Shape: (40,)
```
**What is MFCC?** A 40-dimensional feature vector representing the audio signal in a way that mimics human auditory perception. The most widely used feature set for speech recognition and audio classification.

### 2. Data Preparation
```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y_encoded = le.fit_transform(labels)  # 8 duygu → 0-7

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded,
                                                      test_size=0.2, random_state=42)
```

### 3. SVM Pipeline with GridSearchCV
```python
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(probability=True))
])

param_grid = {
    'svm__C':      [0.1, 1, 10, 100],
    'svm__gamma':  ['scale', 'auto', 0.001, 0.01],
    'svm__kernel': ['rbf', 'linear']
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5,
                            scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
```
> Using StandardScaler inside a Pipeline prevents data leakage during cross-validation.

### 4. Evaluation
```python
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

y_pred = best_model.predict(X_test)
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm, display_labels=le.classes_).plot()
```

### 5. Model Serialization + Real-time Prediction
```python
import joblib
joblib.dump(best_model, 'emotion_model.joblib')

def predict_emotion(file_path):
    features = extract_features(file_path)
    features = np.array(features).reshape(1, -1)
    prediction = best_model.predict(features)
    return le.inverse_transform(prediction)[0]

# Gerçek ses dosyaları üzerinde test
predict_emotion('/content/drive/MyDrive/aa.wav')
predict_emotion('/content/drive/MyDrive/angry.wav')
```

---

## 📦 Libraries Used
```python
pandas, numpy, librosa, joblib, matplotlib,
sklearn (SVC, Pipeline, StandardScaler, LabelEncoder, GridSearchCV,
train_test_split, classification_report, confusion_matrix, ConfusionMatrixDisplay)
```

---

## 💡 Key Takeaways
- MFCC (40 coefficients) → represents audio in frequency bands that mirror human auditory perception; the strongest feature set for speech tasks
- SVM is a strong baseline for small-to-medium audio datasets — performs well in high-dimensional MFCC space
- Pipeline + GridSearchCV combination → prevents data leakage AND finds optimal hyperparameters
- `probability=True` → enables SVC's predict_proba support, allowing confidence scores per class
- Model saved with `joblib` can be deployed on new `.wav` files — a real-world inference pipeline
- This is the only audio-based ML project in the portfolio — going beyond tabular data
