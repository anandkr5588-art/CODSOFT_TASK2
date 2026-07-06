# 💳 Credit Card Fraud Detection

## 📌 Project Overview

This project is developed as **Task 2** for the **CodSoft Machine Learning Internship**.

The objective of this project is to build a Machine Learning model that detects fraudulent credit card transactions using transaction details. The model is trained to classify transactions as either **Fraudulent** or **Legitimate**.

---

## 📂 Repository Name

**CODSOFT_TASK2**

---

## 🎯 Objective

To develop a machine learning model that can accurately identify fraudulent credit card transactions and help reduce financial fraud.

---

## 📊 Dataset

**Dataset Used:** Fraud Detection Dataset

**Kaggle Dataset Link:**
https://www.kaggle.com/datasets/kartik2112/fraud-detection

### Dataset File

The project uses the following dataset:

* `fraudTrain.csv`

> **Note:** The dataset is **not included** in this repository because of its large file size. Download the dataset from the Kaggle link above and place **`fraudTrain.csv`** inside the **DATASET** folder (or the project root) before running the program.

---

## 📁 Project Structure

```text
CODSOFT_TASK2
│
├── DATASET
│   └── fraudTrain.csv
│
├── credit_card_fraud_detection.py
├── requirements.txt
├── README.md
└── description.txt
```

---

## 🛠 Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Joblib

---

## 📚 Machine Learning Workflow

* Load the dataset
* Perform data preprocessing
* Handle missing values
* Encode categorical features
* Split the dataset into training and testing sets
* Train the Random Forest Classifier
* Predict fraudulent transactions
* Evaluate the model using Accuracy Score, Confusion Matrix, and Classification Report
* Save the trained model

---

## 🤖 Machine Learning Model

### Classification Algorithm

* Random Forest Classifier

---

## 📈 Model Performance

The trained model is evaluated using:

* Accuracy Score
* Confusion Matrix
* Precision
* Recall
* F1-Score
* Classification Report

> **Note:** The accuracy may vary slightly depending on the dataset version and train-test split.

---

## 🎯 Sample Prediction

### Input

A credit card transaction with customer and transaction details.

### Predicted Output

* Legitimate Transaction (0)
* Fraudulent Transaction (1)

---

## ▶️ How to Run the Project

### Step 1

Clone the repository:

```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/CODSOFT_TASK2.git
```

### Step 2

Go to the project folder:

```bash
cd CODSOFT_TASK2
```

### Step 3

Install the required libraries:

```bash
pip install -r requirements.txt
```

### Step 4

Download the dataset from Kaggle and place **`fraudTrain.csv`** inside the **DATASET** folder.

### Step 5

Run the program:

```bash
python credit_card_fraud_detection.py
```

---

## 📋 Requirements

* pandas
* numpy
* scikit-learn
* joblib

---

## 📁 Project Files

* `credit_card_fraud_detection.py`
* `requirements.txt`
* `README.md`
* `description.txt`
* `DATASET/`

---

## 🚀 Future Improvements

* Train using XGBoost and LightGBM
* Handle class imbalance using SMOTE
* Perform hyperparameter tuning
* Build a real-time fraud detection system
* Deploy the model as a web application using Flask or Streamlit

---

## 📸 Output

The program displays:

* Dataset information
* Model Accuracy
* Confusion Matrix
* Classification Report

It also saves the trained model as:

```text
credit_card_fraud_model.pkl
```

---

## 👨‍💻 Author

**Anand Kumar**

B.Tech (Artificial Intelligence & Machine Learning)

CodSoft Machine Learning Intern
