# Week 4 – Advanced ML: Customer Churn Prediction

**Intern:** Ali Akaash | **Supervisor:** Qamar Naveed | **Due:** 3 July 2026

---

## 📌 Project Overview
An end-to-end Machine Learning project that predicts customer churn using advanced ML techniques, hyperparameter tuning, cross-validation, and an interactive Streamlit dashboard.

**Dataset:** [Telco Customer Churn – Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

---

## 🗂️ Project Structure
```
Week4-Advanced-ML-Optimization/
│
├── data/                  ← Dataset (CSV)
├── notebooks/             ← Jupyter Notebook (.ipynb)
├── models/
│   ├── best_model.pkl     ← Tuned best model
│   └── scaler.pkl         ← Fitted StandardScaler
├── screenshots/           ← Dashboard screenshots
├── reports/               ← Model comparison report
├── utils/
│   └── helpers.py         ← Reusable functions
├── app.py                 ← Streamlit dashboard
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup Instructions

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/Week4-Advanced-ML-Optimization.git
cd Week4-Advanced-ML-Optimization

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run Jupyter Notebook (Day 1)
jupyter notebook notebooks/

# 4. Run Streamlit App (Day 2)
streamlit run app.py
```

---

## 🤖 Models Trained
| Model | Accuracy | F1-Score | ROC-AUC |
|---|---|---|---|
| Logistic Regression | — | — | — |
| Decision Tree | — | — | — |
| Random Forest ⭐ | — | — | — |
| KNN | — | — | — |
| SVM | — | — | — |

*(Filled after training)*

---

## 📊 Dashboard Pages
- 🏠 Home
- 📊 Dataset Overview
- 📈 EDA (Heatmap, Histograms, Class Distribution)
- 🤖 Model Comparison
- 🎯 Prediction Form + CSV Download
- ⭐ Feature Importance
- ℹ About

---

## 🔧 Techniques Used
- Feature Engineering & Selection
- GridSearchCV + RandomizedSearchCV
- Stratified K-Fold Cross Validation
- Model Persistence (Joblib)

---

## 📁 Deliverables
- [x] Jupyter Notebook
- [x] Trained & Tuned Model (`best_model.pkl`)
- [x] Streamlit Application (`app.py`)
- [x] Saved Model File
- [x] GitHub Repository
- [x] README Documentation
- [ ] Screenshots *(add after Day 2)*
- [ ] Model Comparison Report *(add after Day 1)*
