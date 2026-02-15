# 🚀 Employee Attrition Risk Analysis & Prediction

Employee attrition is a critical challenge for Indian organizations. High turnover leads to increased recruitment costs, productivity losses, and the departure of experienced talent. This project leverages Machine Learning to identify high-risk employees early, enabling data-driven retention strategies.

---

## 📊 Dataset Insights
The analysis is based on a dataset of **1,470 records** with **35 unique features**. 

### **Key Statistics:**
* **Average Age**: 36.9 years
* **Average Monthly Income**: ₹5,46,168 (approx. converted to INR)
* **Average Experience**: 11.3 total working years
* **Attrition Rate**: 16.1% (Market standard for mid-sized firms)

### **Critical Observations:**
* **Demographics**: Most employees are between **30–40 years** old.
* **Income Correlation**: Monthly income shows a strong positive correlation with professional experience.
* **Class Imbalance**: The dataset is imbalanced, with the majority of employees staying (Class 0), requiring specific evaluation metrics beyond simple accuracy.

---

## ⚙️ Project Workflow
1.  **Data Exploration**: Analyzing features like `Age`, `OverTime`, `JobRole`, and `MonthlyIncome`.
2.  **Preprocessing**: Implementing `LabelEncoder` for categorical variables and `StandardScaler` for feature scaling.
3.  **Model Selection**: Training and evaluating multiple classifiers to detect the minority class (employees leaving).



[Image of machine learning model training workflow]


---

## 🏆 Model Performance
Given the class imbalance, the models were evaluated primarily on **Recall** and **F1-Score** for Class 1 (Attrition).

| Model | Precision (Class 1) | Recall (Class 1) | F1-Score |
| :--- | :---: | :---: | :---: |
| **XGBoost** | **0.50** | **0.41** | **0.46** |
| **Logistic Regression** | 0.50 | 0.36 | 0.42 |
| **Random Forest** | 1.00 | 0.13 | 0.23 |

### **Key Findings:**
* **XGBoost** emerged as the top performer, offering the best balance for detecting the minority class.
* **Random Forest** achieved perfect precision (1.00) but failed on recall (0.13), meaning it missed 87% of employees who actually left.
* **Metric Focus**: Accuracy is ignored as a success metric here because the "Stayed" class dominates the data.

--

## 🛠️ Tech Stack
* **Language**: Python 🐍
* **Data Handling**: `Pandas`, `NumPy`
* **Visualization**: `Matplotlib`, `Seaborn`
* **Machine Learning**: `Scikit-learn`, `XGBoost`,`Logistic Regression`, `Random Forest`

---

## 💡 Recommendations & Future Scope
To further improve performance in the Indian corporate context:
1.  **SMOTE (Oversampling)**: Use synthetic data generation to balance the attrition classes.
2.  **Threshold Tuning**: Lower the classification threshold to prioritize **Recall** (catching more potential leavers).
3.  **Feature Engineering**: Include factors like "Distance from Home" and "Years Since Last Promotion" which are high-impact variables in Indian urban employment.

---


