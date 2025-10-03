# Early Cancer Prediction using Machine Learning

## Overview
Breast cancer is one of the most common cancers affecting women worldwide. Early detection can save lives, and machine learning can assist in identifying the risk based on diagnostic data.

This project uses **machine learning** and **deep learning** models to predict whether a tumor is **malignant (M) or benign (B)**. It’s built in Python using **scikit-learn** and **TensorFlow**, with a focus on clarity, usability, and real-world applicability.

---

## Project Goal
- Predict cancer diagnosis based on tumor features.
- Compare **Logistic Regression** and a **Neural Network** for performance.
- Highlight the **most important features** affecting predictions.
- Save trained models for easy deployment or future use.

---

## Dataset
- Source: [Breast Cancer Wisconsin (Diagnostic) Dataset on Kaggle](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)
- **Features:** 30 numeric attributes describing tumor characteristics (radius, texture, smoothness, etc.)
- **Target:** `diagnosis` → `M` (Malignant) or `B` (Benign)

> Dataset is included as `data/data.csv`. You can also download it from Kaggle if needed.

---

## How It Works
1. **Load & Preprocess Data**
   - Map `diagnosis` to 0 (Benign) and 1 (Malignant)
   - Handle missing values with imputation
   - Scale features for model training

2. **Split Data**
   - 80% training, 20% testing

3. **Train Models**
   - **Logistic Regression** with hyperparameter tuning
   - **Neural Network** with two hidden layers for comparison

4. **Evaluate Models**
   - Accuracy, precision, recall, F1-score
   - Confusion matrix
   - ROC curve and AUC score

5. **Analyze Features**
   - Identify the **top features** influencing predictions

6. **Save Models**
   - `logistic_model.pkl` & `scaler.pkl` for Logistic Regression
   - `nn_model.h5` for Neural Network

---

## Results

| Model               | Test Accuracy | ROC-AUC Score |
|--------------------|---------------|---------------|
| Logistic Regression | ~90%          | 0.96          |
| Neural Network      | ~92%          | 0.97          |

**Top 10 Important Features:**
- `worst concave points`
- `worst perimeter`
- `mean concave points`
- `worst radius`
- `mean radius`
- `mean texture`
- `worst area`
- `mean concavity`
- `worst compactness`
- `mean perimeter`

> ROC curves and evaluation plots are included in the `screenshots/` folder.

---

## How to Use

```python
import joblib
from tensorflow.keras.models import load_model

# Load Logistic Regression
log_model = joblib.load('models/logistic_model.pkl')
scaler = joblib.load('models/scaler.pkl')

# Load Neural Network
nn_model = load_model('models/nn_model.h5')

# Predict on new data
X_new_scaled = scaler.transform(X_new)  # scale features
y_pred_lr = log_model.predict(X_new_scaled)  # Logistic Regression prediction
y_pred_nn = (nn_model.predict(X_new_scaled) > 0.5).astype(int)  # Neural Network prediction

Future Improvements
Explore other machine learning models (Random Forest, XGBoost, SVM)
Deploy as a web application for real-time cancer prediction
Add feature visualization to explain model decisions to medical professionals
Expand dataset for more robust performance

Why This Project Matters
This project demonstrates how AI can assist in preventive healthcare, helping doctors and patients identify cancer risk early. Early detection can significantly improve treatment outcomes and potentially save lives.
