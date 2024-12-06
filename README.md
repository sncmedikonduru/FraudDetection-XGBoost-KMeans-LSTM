
# **Credit Card Fraud Detection Project**

This project is focused on detecting fraudulent transactions using machine learning techniques. It employs a variety of models such as KNN, XGBoost, and LSTM, combined using an **average ensembling method** to achieve robust performance on the highly imbalanced credit card fraud dataset.

---

## **Project Overview**
Fraud detection is a critical task in financial systems, where identifying fraudulent transactions accurately is essential to prevent monetary losses and customer dissatisfaction. This project utilizes the publicly available **Kaggle Credit Card Fraud Detection dataset**, which is highly imbalanced, to implement a robust fraud detection system.

---

## **Dataset**
Dataset available at https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
The dataset contains credit card transactions made by European cardholders in September 2013. It is highly imbalanced, with only 0.17% of transactions classified as fraudulent.

- **Total Records**: 284,807
- **Fraud Cases (Class = 1)**: 492
- **Non-Fraud Cases (Class = 0)**: 284,315

### **Key Features**:
- `Time`: Seconds elapsed between the first transaction and each subsequent transaction.
- `Amount`: Transaction amount.
- `Class`: Target variable (0 = Non-Fraud, 1 = Fraud).
- `V1` to `V28`: Principal components obtained from PCA (features are anonymized).

---

## **Workflow**
1. **Data Preprocessing**:
   - Handled missing values and duplicates.
   - Standardized numerical features using `StandardScaler`.
2. **Class Balancing**:
   - Used **SMOTEENN** (SMOTE + Edited Nearest Neighbors) to balance the dataset.
3. **Exploratory Data Analysis (EDA)**:
   - Analyzed class distributions, feature correlations, and feature distributions.
4. **Model Training**:
   - Trained individual models: **KNN**, **XGBoost**, and **LSTM**.
5. **Model Ensembling**:
   - Combined predictions using **average ensembling** with equal weights for all models.
6. **Evaluation**:
   - Evaluated performance using metrics like **Precision**, **Recall**, **F1-Score**, and **Confusion Matrix**.

---

## **Installation**

### **Clone the Repository**
```bash
git clone https://github.com/<your-username>/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```

### **Environment Setup**
1. Create a virtual environment:
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## **How to Run**
1. **Run the Jupyter Notebook**:
   - Open the `fraud_detection.ipynb` file in Jupyter Notebook or Google Colab.
   - Follow the step-by-step implementation to preprocess data, train models, and evaluate results.

2. **Run the Python Script**:
   - Execute `fraud_detection.py` for a streamlined version of the project:
     ```bash
     python fraud_detection.py
     ```

---

## **Technologies Used**
- **Programming Language**: Python
- **Libraries**:
  - `pandas`, `numpy`, `matplotlib`, `seaborn`
  - `scikit-learn`, `imblearn`, `xgboost`
  - `tensorflow` (for LSTM)
