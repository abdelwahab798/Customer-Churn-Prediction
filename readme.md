# 🚀 Customer Churn Prediction using XGBoost & Streamlit

This project focuses on predicting **customer churn** for a telecom company using **XGBoost Classifier**.  
It includes end-to-end steps from data preprocessing and model training to deployment with **Streamlit** for real-time predictions.

---

## 📋 Project Summary

The goal of this project is to help telecom companies identify customers who are likely to leave (churn) and take proactive measures to retain them.  
By applying machine learning, we can detect hidden patterns in customer behavior and improve business decisions.

---

## 🔍 1. Data Understanding & Preprocessing

- The dataset contains customer information such as contract type, payment method, service duration, and billing details.  
- Handled missing and inconsistent values.  
- Converted categorical features to numeric using **Label Encoding**.  
- Created a new engineered feature `service_plus`, representing the total number of services used by each customer (OnlineSecurity, Backup, TechSupport, etc.).  
- Removed duplicates and corrected data types (e.g., `TotalCharges` → numeric).  

---

## 🧠 2. Model Building & Evaluation

Tested multiple classification models:
- **Logistic Regression** → Accuracy: **73%**
- **XGBRFClassifier (XGBoost Random Forest)** → Accuracy: **77%**
- **XGBoost Classifier (Tuned)** → Best overall performance after fine-tuning hyperparameters.

### ✅ Best Model: XGBoost (Tuned)
| Metric | Score |
|--------|--------|
| Accuracy | 79.6% |
| Recall (Churned Customers) | 78% |
| Precision | 49% |

To handle data imbalance, `scale_pos_weight` was applied to improve the model’s ability to detect churned customers.

---

## ⚙️ 3. Hyperparameter Tuning

Optimized using **GridSearchCV**, achieving the best cross-validation accuracy (**0.8087**) with the following parameters:

```python
{'colsample_bytree': 0.8,
 'gamma': 0,
 'learning_rate': 0.05,
 'max_depth': 3,
 'n_estimators': 300,
 'subsample': 1.0}
 

```

# 4. 💡 Insights

### Key insights derived from the data:

- 📉 Customers with **month-to-month contracts** and **electronic payment methods** are more likely to churn.  
- ⏳ A **longer tenure** significantly decreases the likelihood of churn.  
- 🔒 Customers subscribed to **additional services** (Online Security, Backup, Tech Support) tend to be more loyal.  

---

# 5. 🌐 Deployment

- The final trained model is saved as **`model_xgboost.pkl`**.  
- An **interactive Streamlit app** was built to predict churn probability based on customer details.  

### The app displays:
- ✅ **"Customer likely to stay"**  
- 🚨 **"Customer likely to churn"**

---

# 6. 🧰 Tools & Technologies

| **Category** | **Tools** |
|---------------|-----------|
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Machine Learning** | Scikit-learn, XGBoost |
| **Model Tuning** | GridSearchCV |
| **Deployment** | Streamlit |
| **Environment** | Conda (`tens`) |

---

# 7. 📈 Results Summary

- Achieved **balanced performance** between accuracy and recall.  
- Built a **robust model** capable of detecting high-risk churn customers.  
- Delivered a **fully functional ML web app** ready for real-world use by customer retention teams.  

---

#  8. 📦 Folder Structure

```plaintext
Customer-Churn-Prediction/
│
├── data/
│   |_ Orginal Data.csv
|    |_ clean Data.csv
|    |_ model Data.csv
│
├── notebooks/
│   |_ data preprocessing.ipynb
|    |_ visual.ipynb
|    |_ xg_model.ipynb
|    |_ log_model.ipynb
│
├── model/
│   |_ model_xgboost.pkl
│
├── deployment/
│   |_ app.py
│
├── README.md
└── requirements.txt
```
# 9.🏁 Conclusion
This project demonstrates how machine learning can effectively predict and understand customer churn.
Through an interactive Streamlit interface, it bridges the gap between data science insights and business decision-making — turning predictions into actionable strategies for customer retention.
