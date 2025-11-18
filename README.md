# 🫀 Heart Attack Prediction Using Machine Learning

# 📌 Overview  
This project focuses on predicting the risk of heart disease using supervised machine learning algorithms on structured health data. The goal is to build a reliable predictive model that could assist healthcare professionals in early diagnosis and risk assessment.

---

# 📂 Project Structure
```
├── data/                 # Dataset used
├── models/               # Saved models (.pkl)
├── notebooks/            # Jupyter Notebooks
├── app.py                # Streamlit app (for deployment)
├── requirements.txt      # Dependencies
└── README.md
```

---

# 🧠 Problem Statement  
Heart disease is a leading cause of death worldwide. Early prediction can help prevent fatal incidents. This project builds a machine learning model to classify whether an individual is at risk of a heart attack based on various health indicators.

---

# 📁 Dataset

The dataset contains *8,763* user records and *26 features*, sourced from [Kaggle](https://share.google/wOcbIlE09Nyk0fOzG). It includes both demographic and medical information used to predict heart attack risk.

# Key Features:
- *Patient ID:* Unique identifier for each patient.

- *Age:* Age of the patient.

- *Sex:* Gender of the patient (Male/Female).

- *Cholesterol:* Cholesterol levels of the patient.

- *Blood Pressure:* Blood pressure of the patient (systolic/diastolic).

- Heart Rate: Heart rate of the patient.

- Diabetes: Whether the patient has diabetes (Yes/No).

- Family History: Family history of heart-related problems (1: Yes, 0: No).

- Smoking: Smoking status of the patient (1: Smoker, 0: Non-smoker).

- Obesity: Obesity status of the patient (1: Obese, 0: Not obese).

- Alcohol Consumption: Level of alcohol consumption by the patient (None/Light/Moderate/Heavy).

- Exercise Hours Per Week: Number of exercise hours per week.

- Diet: Dietary habits of the patient (Healthy/Average/Unhealthy).

- Previous Heart Problems: Previous heart problems of the patient (1: Yes, 0: No).

- Medication Use: Medication usage by the patient (1: Yes, 0: No).

- Stress Level: Stress level reported by the patient (1-10).

- Sedentary Hours Per Day: Hours of sedentary activity per day.

- Income: Income level of the patient.

- BMI: Body Mass Index (BMI) of the patient.

- Triglycerides: Triglyceride levels of the patient.

- Physical Activity Days Per Week: Days of physical activity per week.

- Sleep Hours Per Day: Hours of sleep per day.

- Country: Country of the patient.

- Continent: Continent where the patient resides.

- Hemisphere: Hemisphere where the patient resides.

- Heart Attack Risk (Outcome): Presence of heart attack risk (1: Yes, 0: No).

-----

# 🧼 Data Preprocessing  

- Handled missing values  
- Applied feature scaling (`StandardScaler`)  
- Removed irrelevant features (e.g., `Patient ID`, `Hemisphere`, `Continent`, `Blood Pressure`)  
-----

# 📊 Data Visualizations  
- KDE plots, boxplots, histograms  
- Correlation matrix heatmap  
- Count plots for categorical features 

-----

# 📊 Exploratory Data Analysis (EDA)
- Distribution analysis (Age, Cholesterol, BMI, etc.)  
- Correlation heatmap to identify relationships  
- Outlier detection (boxplots)  
- Class distribution  

---

# ⚙️ Feature Engineering  
Created interaction and ratio-based features to enhance model performance:
- `BMI_Stress` = BMI × Stress Level  
- `Activity_Ratio` = Exercise Hours per Week / (Sedentary Hours Per Day + 1)  
- `BP_Product` = Systolic_BP × Diastolic_BP  
- `Sleep_Stress_Interaction`  = Sleep Hours Per Day × Stress Level
- `Substance_Use` = Smoking * Alcohol Consumption
- `Diastolic_BP`
- `Systolic_BP`

-------

# 🧪 Feature Selection
   - Methods like:
     - *RFE (Recursive Feature Elimination)*
     - *L1_Lasso*
     - *SelectKBest*
     - *Mutual Information*
     - *Random Forest*
    
 ---

# ⚖️ SMOTE (Synthetic Oversampling)  
- Balanced the dataset using SMOTE to address class imbalance  
- Improved model generalization and recall on minority class  

---

# 🔍 Model Building & Hyperparameter Tuning  
Models trained with `GridSearchCV` with `StratifiedKFold` (k = 5) cross-validation:  
- Logistic Regression  
- Random Forest  
- LightGBM  
- CatBoost  
- MLP (Neural Network)  

---

# 📈  Model Evaluation  
Evaluated model using:  
- Accuracy  
- Precision  
- Recall
- F1_Score

---



# 🧰 Tech Stack

- *Language*: Python
- *Libraries*: Pandas, NumPy, Scikit-learn, XGBoost, CatBoost, imbalanced-learn (SMOTE), Matplotlib, Seabornu
- *Tools*: Jupyter Notebook

-----

# 📈 Results & Observations

After training multiple ML models with various feature selection methods, here are the top results:

| Feature Selection | Model               | Accuracy | Precision | Recall  | F1 Score |
|------------------|---------------------|----------|-----------|---------|----------|
| L1_Lasso         | Logistic Regression | 0.692889 | 0.800554  | 0.513778 | 0.625880 |
| Mutual_Info      | XGBoost             | 0.644889 | 0.647378  | 0.636444 | 0.641865 |
| SelectKBest      | CatBoost            | 0.640289 | 0.679502  | 0.533333 | 0.597610 |

# 📌 Key Observations:

- *Logistic Regression + L1_Lasso* gave the best F1 Score, showing balanced performance between precision and recall.
- *XGBoost + Mutual_Info* had the best recall and strong overall balance, making it ideal where reducing false negatives is critical.
- *CatBoost + SelectKBest* showed the highest precision, which is useful when false positives need to be minimized.
-----

# ✅ Conclusion  
The trained models effectively predict heart attack risk, supporting medical professionals with early warning tools for diagnosis and preventive measures.

---

# 🚀 Future Improvements  
- Use larger and more diverse datasets  
- Incorporate clinical test results (e.g., ECG, CT scans)  
- Apply deep learning techniques (e.g., CNN on medical images)  
- Integrate into healthcare platforms for real-time predictions  

---
# 🔗 Links

- *GitHub Repo*: [Insert your GitHub repo link]


