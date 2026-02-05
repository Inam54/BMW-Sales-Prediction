# 🚗 BMW Sales Classification System

## 📄 Project Overview

This project aims to **classify BMW car sales performance** as **High** or **Low** based on vehicle specifications, pricing, region, and other categorical attributes.

It demonstrates an **industry-style machine learning workflow** using:
- Pipelines
- ColumnTransformer
- Multiple models for comparison
- Proper preprocessing for numerical and categorical data

The project is implemented using **Object-Oriented Programming (OOP)** in Python.

---

## 🗂 Dataset

**Dataset:** BMW Sales Dataset  
**File:** `Dataset/BMW Sales.csv`

### Description
The dataset contains information related to BMW car sales, including:
- Vehicle specifications (engine size, mileage)
- Pricing information
- Categorical attributes (model, region, fuel type, color, transmission)
- Sales volume and sales classification

**Target Variable:**  
`Sales_Classification`
- `Low` → 0  
- `High` → 1  

---

## 🧰 Project Files

- `bmw_sales_prediction.py` – Main Python script implementing ML models using OOP
- `analysis.ipynb` – Exploratory Data Analysis and preprocessing decisions
- `Dataset/BMW Sales.csv` – Dataset (excluded from GitHub via `.gitignore`)
- `requirements.txt` – Python dependencies
- `README.md` – Project documentation

---

## 🔧 Key Techniques Used

### Data Preprocessing
- **StandardScaler** for numerical features:
  - `Mileage_KM`
  - `Price_USD`
- **RobustScaler** for outlier-sensitive feature:
  - `Engine_Size_L`
- **One-Hot Encoding** for categorical features:
  - `Model`
  - `Region`
  - `Fuel_Type`
  - `Color`
  - `Transmission`
- Target encoding for binary classification (`Low`, `High`)

---

## 🧠 Machine Learning Models

The following models are implemented and evaluated:

### 1️⃣ Logistic Regression
- Used as a baseline classifier
- Handles class imbalance using `class_weight="balanced"`
- Implemented inside a full preprocessing pipeline

### 2️⃣ Decision Tree Classifier
- Tuned hyperparameters:
  - `max_depth`
  - `min_samples_leaf`
  - `min_samples_split`
- Balanced class weights to handle imbalance

### 3️⃣ Random Forest Classifier
- Ensemble-based model
- Key parameters:
  - `n_estimators = 500`
  - `max_depth = 8`
  - `max_features = sqrt`
  - `class_weight = balanced`

---

## 📊 Model Evaluation

Each model is evaluated using:

- **Confusion Matrix**
- **Classification Report**, including:
  - Precision
  - Recall
  - F1-score
  - Support

Evaluation is performed on a **hold-out test set (80/20 split)**.

---

## 🏗️ Pipeline Design

- `ColumnTransformer` used for handling mixed data types
- Separate pipelines for:
  - Numerical scaling
  - Categorical encoding
- Final pipeline structure:
Preprocessing → Model


This design ensures:
- Clean code
- Reproducibility
- Industry-standard ML workflow

---

## 💡 Learning Outcomes

- Practical use of `Pipeline` and `ColumnTransformer`
- Handling categorical and numerical features correctly
- Implementing multiple models in a single OOP-based project
- Understanding precision–recall trade-offs in classification
- Working with imbalanced datasets using class weights

---

## 🚀 Future Improvements

- Hyperparameter tuning using GridSearchCV
- Cross-validation for robust evaluation
- Feature importance analysis
- Threshold tuning for business-specific objectives
- Deployment using Flask or FastAPI

---

## 🛠 Tech Stack

- Python
- pandas
- scikit-learn
- matplotlib
- seaborn
- Jupyter Notebook

---

## 📁 Project Structure

BMW-Sales-Classification/
├── Dataset/
│ └── BMW Sales.csv
├── analysis.ipynb
├── bmw_sales_prediction.py
├── requirements.txt
└── README.md


---

## 👤 Author

**Inam Ur Rehman**  
BS Computer Engineering (ITU)  
Focus: Machine Learning | Deep Learning | AI Engineering