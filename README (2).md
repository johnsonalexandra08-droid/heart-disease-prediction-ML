# Heart Disease Prediction Using Machine Learning

**Student:** Hannah Johnson  
**Course:** 202602 AAI201 26214 – Machine Learning  
**Assignment:** M15: Capstone Project – Designing a Machine Learning Solution  
**Due Date:** May 5, 2026

---

## Project Overview

This capstone project builds and evaluates a **binary classification machine learning model** to predict whether a patient has heart disease based on 13 routine clinical measurements. The project uses the **UCI Heart Disease Dataset (Cleveland subset)** — one of the most widely cited benchmarks in medical ML.

Heart disease is the leading cause of death in the United States, accounting for approximately 1 in 5 deaths annually. Early detection from routine clinical data could help clinicians prioritize high-risk patients and reduce expensive diagnostic workups.

### Problem Statement
> *Can a machine learning model trained on routine clinical measurements reliably predict heart disease diagnosis with high accuracy, precision, and recall?*

---

## Dataset

**Source:** UCI Machine Learning Repository — Heart Disease Dataset  
**URL:** https://archive.ics.uci.edu/dataset/45/heart+disease  
**Also available:** https://www.kaggle.com/datasets/ronitf/heart-disease-uci  

**Features (13 clinical variables):**

| Feature | Description |
|---|---|
| `age` | Age in years |
| `sex` | 1 = Male, 0 = Female |
| `cp` | Chest pain type (0–3) |
| `trestbps` | Resting blood pressure (mmHg) |
| `chol` | Serum cholesterol (mg/dl) |
| `fbs` | Fasting blood sugar > 120 mg/dl |
| `restecg` | Resting ECG results (0–2) |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise-induced angina |
| `oldpeak` | ST depression from exercise |
| `slope` | Slope of peak exercise ST segment |
| `ca` | Major vessels colored by fluoroscopy |
| `thal` | Thalassemia type |

**Target:** Binary — 0 = No Disease, 1 = Heart Disease  
**Samples:** 303 (297 after missing value removal)

---

## Models Evaluated

| Model | Validation F1 | Validation AUC |
|---|---|---|
| Logistic Regression | ~0.84 | ~0.92 |
| Naïve Bayes | ~0.80 | ~0.87 |
| Decision Tree | ~0.81 | ~0.84 |
| Random Forest | ~0.87 | ~0.93 |
| MLP Neural Network | ~0.87 | ~0.93 |

**Best model:** Random Forest / MLP (comparable performance; Random Forest preferred for interpretability)

---

## Overfitting Prevention Techniques

1. **L2 Regularization** — Applied to Logistic Regression (`C=1.0`) and MLP (`alpha=0.01`)
2. **Early Stopping** — MLP halts when validation loss stops improving (patience=15)
3. **Tree Depth Limiting** — Decision Tree (`max_depth=5`), Random Forest (`max_depth=8`)
4. **Ensemble Averaging** — Random Forest with 200 trees reduces variance
5. **5-Fold Cross-Validation** — Used for unbiased generalization estimates

---

## Bias & Fairness

- **Sex imbalance identified:** ~68% male, ~32% female patients
- **SMOTE applied** to training set to balance classes
- **Class weights** set to `'balanced'` in all applicable models
- **Subgroup evaluation** performed separately for male and female patients

---

## Setup Instructions

### Requirements

```bash
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn jupyter
```

### Run the Notebook

```bash
git clone https://github.com/your-username/heart-disease-prediction
cd heart-disease-prediction
jupyter notebook HeartDiseasePrediction_HannahJohnson.ipynb
```

Or open in Google Colab:  
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/your-username/heart-disease-prediction/blob/main/HeartDiseasePrediction_HannahJohnson.ipynb)

---

## Repository Structure

```
heart-disease-prediction/
├── HeartDiseasePrediction_HannahJohnson.ipynb   # Main notebook
├── HeartDiseasePrediction_Presentation.pptx     # 7-slide presentation
├── README.md                                     # This file
└── outputs/
    ├── class_distribution.png
    ├── feature_distributions.png
    ├── correlation_matrix.png
    ├── bias_analysis.png
    ├── model_comparison.png
    ├── roc_curves.png
    ├── final_evaluation.png
    └── feature_importance.png
```

---

## Presentation

The project presentation summarizes the problem, methodology, results, and key takeaways in 7 slides.

**File:** `HeartDiseasePrediction_Presentation.pptx`

---

## Key Results

- Best model achieved **~87% F1-score** and **~0.93 AUC-ROC** on held-out test data
- Top predictive features: chest pain type (`cp`), max heart rate (`thalach`), fluoroscopy vessels (`ca`), ST depression (`oldpeak`), thalassemia type (`thal`)
- Model correctly discriminates heart disease vs. no disease **93% of the time** (AUC)
- Subgroup analysis confirms model performs comparably across sex groups after bias mitigation

---

## References

- Detrano, R. et al. (1989). International application of a new probability algorithm for the diagnosis of coronary artery disease. *American Journal of Cardiology*, 64(5), 304-310.
- UCI Machine Learning Repository: https://archive.ics.uci.edu/dataset/45/heart+disease
- Chawla, N. V. et al. (2002). SMOTE: Synthetic Minority Over-sampling Technique. *JMLR*, 3, 321-357.

---

*Hannah Johnson | AAI201 Machine Learning | May 2026*
