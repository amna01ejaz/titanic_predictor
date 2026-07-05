# 🚢 Titanic Survival Prediction

This project implements a complete end-to-end machine learning pipeline to predict passenger survival outcomes from the historic Titanic disaster. It covers data preprocessing, missing value imputation, advanced feature engineering, and model optimization.

---

## 📈 Model Performance & Evolution

We started with a simple baseline model and successfully scaled it to a tree-based ensemble method to capture non-linear relationships in passenger demographics.

| Model Algorithm | Accuracy Score | Key Highlights |
| :--- | :--- | :--- |
| **Logistic Regression** | `81.01%` | Baseline implementation; excellent classification stability on Class 0. |
| **Random Forest Classifier** | *[Insert your RF score here]*% | Advanced ensemble model incorporating custom financial and family metrics. |

---

## 🛠️ Feature Engineering Architecture

To extract maximum predictive power from the raw data, the following pipelines were built:
* **Imputation:** Automated median-filling for missing `Age` tracking and mode-filling for missing `Embarked` points.
* **Dimensionality Reduction:** Dropped sparse tracking columns (`Cabin`, `Ticket`, `Name`).
* **Feature Creation:** Created `FamilySize` (combining sibling/parent data) and engineered a custom `FarePerPerson` wealth metric.

---

## 📁 Project Structure

```text
titanic_predictor/
│
├── titanic/
│   ├── train.csv                # Primary training dataset from Kaggle
│   └── test.csv                 # Target evaluation features
│
├── app.py                       # Main production ML pipeline script
├── requirements.txt             # Python environment dependencies
├── .gitignore                   # Version control system exclusions
└── README.md                    # Project documentation