— Loan Default Prediction Project
Loan Default Prediction Using Machine Learning

This project builds a machine learning system that predicts whether a loan applicant will default or successfully repay a loan.
Multiple ML models were trained, evaluated, tuned, and compared to select the best-performing model.

🚀 Project Objective

To analyze loan applicant data and build a predictive model that identifies high-risk loan defaulters, helping financial institutions make better lending decisions.

📊 Dataset Overview

The dataset includes applicant demographics, financial information, credit history, and loan characteristics.

Features Used

Applicant Info

person_age

person_gender

person_education

person_income

person_emp_exp

person_home_ownership

Loan Details

loan_amnt

loan_int_rate

loan_intent

loan_percent_income

Credit History

cb_person_cred_hist_length

credit_score

previous_loan_defaults_on_file

Target Variable

loan_status →
1 = repaid
0 = default

🛠️ Machine Learning Pipeline

The workflow includes the following steps:

1️⃣ Data Preprocessing

Handling missing values

Outlier treatment using IQR capping

Binary & One-Hot encoding

Standardizing and transforming skewed variables

2️⃣ Feature Engineering

New features were created to improve prediction accuracy:

approx_emi → loan burden

balance_income → disposable income

dti → debt-to-income ratio

credit_age_ratio → financial maturity

income_exp_ratio → income stability

Log transforms for normalized distributions

These features significantly improved model performance.

3️⃣ Train-Test Split

80% training, 20% testing

Stratified split to preserve class balance

4️⃣ Models Trained
Model	Category
Logistic Regression	Baseline
KNN	Baseline
Decision Tree	Baseline
Naive Bayes	Baseline
SVM	Advanced
Random Forest	Ensemble
Gradient Boosting	Ensemble
AdaBoost	Ensemble
XGBoost	Advanced Boosting
5️⃣ Hyperparameter Tuning

Performed on:

Random Forest (GridSearchCV)

XGBoost (RandomizedSearchCV)

Using ROC-AUC scoring.

6️⃣ Model Evaluation Metrics

Accuracy

F1 Score

ROC-AUC

Confusion Matrix

ROC Curve

🏆 Best Model Result

The best model was:

🎉 Random Forest (Tuned)

Performance Metrics:

Accuracy: 0.9361

F1 Score: 0.8464

ROC-AUC: 0.97895 (highest among all models)

This model outperformed all others, including XGBoost.

📈 Visualizations

The project includes:

Confusion Matrix (heatmap)

ROC Curve

Side-by-side visualization for easy interpretation

These visuals help understand model behavior and classification quality.

💾 Saving & Loading the Best Model
Saving
import joblib
joblib.dump(best_model, "best_loan_model.pkl")

Loading for Predictions
import joblib
model = joblib.load("best_loan_model.pkl")
prediction = model.predict(new_data)

🧪 Using the Model on New Applicants

Make sure the input data is preprocessed the same way as training data:

prediction = model.predict(applicant_df)
probability = model.predict_proba(applicant_df)[0][1]

📁 Project Structure
Loan-Prediction-Project/
│
├── loan_data.csv
├── processed_loan_data.csv
├── loan_prediction.ipynb
├── best_loan_model.pkl
├── README.md
└── requirements.txt

▶️ How to Run the Project
1. Run the Jupyter notebook
jupyter notebook loan_prediction.ipynb

2. Make predictions

Use best_loan_model.pkl in any Python script or API.

💡 Future Improvements

Add SHAP explainability

Deploy using Flask / FastAPI

Create a Streamlit UI

Handle class imbalance with SMOTE

Try deep learning models