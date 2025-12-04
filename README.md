📘 Loan Default Prediction Using Machine Learning

Predicting whether a loan applicant will default or repay is a critical task for financial institutions.
This project uses multiple machine learning algorithms to build a robust model for identifying high-risk borrowers.

📂 Table of Contents

Overview

Dataset Description

Project Pipeline

Feature Engineering

Models Trained

Model Performance

Best Model

Using the Model

Project Structure

How to Run

Future Improvements

📝 Overview

This project builds a supervised machine learning system to classify loan applicants as:

1 → Loan Repaid

0 → Loan Default

We trained several models, optimized their hyperparameters, evaluated their performance, and selected the best-performing model based on ROC-AUC.

📊 Dataset Description

The dataset includes:

Applicant Features

Age, Gender, Education

Income, Employment Experience

Home Ownership Status

Loan Features

Loan Amount

Interest Rate

Loan Intent (Purpose)

Percent of Income Used for Loan

Credit History

Credit Score

Credit History Length

Previous Defaults (Yes/No)

Target Variable

loan_status → 1 (repaid) / 0 (default)

🔧 Project Pipeline

The machine learning pipeline includes:

1️⃣ Data Cleaning

Missing value handling

Outlier detection & capping (IQR method)

2️⃣ Encoding

Binary encoding → Yes/No

One-hot encoding → Gender, Education, Loan Intent, Home Ownership

3️⃣ Feature Engineering

Powerful new features created to improve prediction quality.

4️⃣ Train–Test Split

80% training

20% testing

Stratified sampling

5️⃣ Model Training

9 machine learning models trained and evaluated.

6️⃣ Hyperparameter Tuning

Performed on:

Random Forest

XGBoost

7️⃣ Model Selection

Chosen based on highest ROC-AUC.

🚀 Feature Engineering

These engineered features significantly improved model accuracy:

Feature	Description
approx_emi	Estimated EMI burden
balance_income	Income after paying EMI
dti	Debt-to-income ratio
credit_age_ratio	Maturity of credit behavior
income_exp_ratio	Income stability measure
log_income / log_loan	Handle skew in numeric features
🤖 Models Trained
Category	Models
Baseline	Logistic Regression, KNN, Decision Tree, Naive Bayes
Advanced	SVM
Ensemble	Random Forest, Gradient Boosting, AdaBoost
Boosting	XGBoost

Each model was evaluated using:

Accuracy

F1 Score

ROC-AUC

Confusion Matrix

ROC Curve

📈 Model Performance Summary

Top-performing models:

Model	Accuracy	F1 Score	ROC-AUC
Random Forest (Tuned)	0.9361	0.8464	0.97895
Random Forest (Base)	0.9369	0.8482	0.97876
XGBoost (Base)	0.9324	0.8382	0.97644
XGBoost (Tuned)	0.9285	0.8279	0.97503
🏆 Best Model
🎉 Random Forest (Tuned)

was selected as the final model because:

Highest ROC-AUC

Highest stability

Best balance between precision & recall

Performs exceptionally well on medium-sized tabular data

Saved as:

best_loan_model.pkl

🧪 Using the Model
Load the Model
import joblib
model = joblib.load("best_loan_model.pkl")

Make Predictions
prediction = model.predict(new_data)
probability = model.predict_proba(new_data)[0][1]


⚠️ New data must follow the same preprocessing steps used during training.

📁 Project Structure
Loan-Prediction-Project/
│
├── data/
│   ├── loan_data.csv
│   ├── processed_loan_data.csv
│
├── models/
│   └── best_loan_model.pkl
│
├── notebooks/
│   └── loan_prediction.ipynb
│
├── README.md
├── requirements.txt

▶️ How to Run
Install dependencies
pip install -r requirements.txt

Open the Notebook
jupyter notebook notebooks/loan_prediction.ipynb

Run all cells

This performs preprocessing, model training, evaluation, and saving the best model.

🔮 Future Improvements

Deploy as a Streamlit web app

Use SHAP for explainability

Handle class imbalance using SMOTE

Add cross-validation for more robust results

Compare with Neural Networks