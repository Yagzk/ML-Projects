# 🎙️ Speech Emotion Recognition — RAVDESS Dataset (SVM + MFCC)

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![Algorithm](https://img.shields.io/badge/Algorithm-SVM-orange) ![Domain](https://img.shields.io/badge/Domain-Audio%20ML-purple) ![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

## 📌 Overview

A **speech emotion recognition** system trained on the RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song) dataset. Audio recordings are converted into numerical feature vectors using **MFCC (Mel-Frequency Cepstral Coefficients)**, and a **Support Vector Machine (SVM)** classifier is trained to recognize the emotional state of the speaker.

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| Name | RAVDESS — Ryerson Audio-Visual Database of Emotional Speech and Song |
| Modality | Audio (WAV files) |
| Emotions | Neutral, Calm, Happy, Sad, Angry, Fearful, Disgust, Surprised |
| Speakers | 24 professional actors (12 male, 12 female) |
| Task | Multi-class Classification |

**Trained model files:**
- `svm_best_model.pkl` — serialized SVM model
- `svm_label_encoder.pkl` — serialized label encoder for emotion classes

---

## 🔧 Methodology

### 1. Audio Feature Extraction — MFCC
MFCCs represent the short-term power spectrum of a sound — they are the standard feature for speech processing:

```python
import librosa
import numpy as np

def extract_features(file_path):
    audio, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    return np.mean(mfccs.T, axis=0)  # Shape: (40,)
```

The result is a **40-dimensional feature vector** per audio file, saved to `features.csv`.

### 2. Label Encoding
Emotion labels (string) → integer codes:
```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_encoded = le.fit_transform(y)
# Saved as: svm_label_encoder.pkl
```

### 3. Train/Test Split + Scaling
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
```

### 4. SVM Classifier
Support Vector Machines are well-suited for high-dimensional audio feature spaces:
```python
from sklearn.svm import SVC
svm = SVC(kernel='rbf', C=..., gamma=...)
svm.fit(X_train_scaled, y_train)
```
Model was saved to `svm_best_model.pkl` for inference on new audio files.

### 5. Real-time Inference
The saved model and encoder allow classifying **any new WAV file**:
```python
import pickle, librosa, numpy as np

# Load model and encoder
with open('svm_best_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('svm_label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

# Extract features from new audio
features = extract_features('new_recording.wav').reshape(1, -1)

# Predict emotion
pred_label = le.inverse_transform(model.predict(features))
print(f"Detected emotion: {pred_label[0]}")
```

---

## 🗂️ Versions

| File | Description |
|------|-------------|
| `ravdess v1.ipynb` | Initial exploration: feature extraction, baseline SVM |
| `ravdess_v2.ipynb` | Refined model: hyperparameter tuning, model serialization, inference pipeline |

---

## 📦 Libraries Used

```python
librosa, numpy, pandas, sklearn (SVC, LabelEncoder, StandardScaler,
train_test_split, classification_report), pickle, matplotlib
```

---

## 💡 Key Takeaways

- **MFCC is the gold standard** for audio feature extraction in speech tasks — it mimics the human ear's frequency response
- SVM performs surprisingly well on MFCC features despite the dataset being relatively small
- Saving both the model AND the label encoder is critical — without the encoder, predictions are uninterpretable integers
- This architecture can be extended to real-time emotion detection in voice assistants, call centers, or mental health monitoring applications
- Version 2 significantly improved over Version 1 by refining preprocessing and adding the inference pipeline
