# 🏠 Predicting House Prices  
*(Still in progress — last updated: 28/01/2026)*

This project focuses on building an **end-to-end machine learning regression pipeline** to predict **median house prices** in California districts using socio-economic and geographic features derived from the **1990 California Census dataset**.

---

## 📊 Dataset Description

The dataset contains information about various housing districts in California.  
Each row represents **aggregated data for a district**, not individual houses.

### 🔑 Key Features

- **Geographical Attributes**
  - Longitude
  - Latitude

- **Housing Attributes**
  - Housing median age
  - Total rooms
  - Total bedrooms

- **Demographic Attributes**
  - Population
  - Households

- **Economic Attribute**
  - Median income

- **Categorical Attribute**
  - Ocean proximity

### 🎯 Target Variable

- **Median House Value** — the value to be predicted

---

## ⚙️ Project Workflow

### 1️⃣ Exploratory Data Analysis (EDA)
- Visualized geographical price trends using latitude & longitude
- Analyzed feature distributions and correlations
- Identified **median income** as a strong predictor of house prices

### 2️⃣ Data Preprocessing
- Handled missing values using imputation techniques
- Applied feature scaling to numerical attributes
- Encoded categorical variables using **one-hot encoding**

### 3️⃣ Feature Engineering
Created meaningful derived features:
- Rooms per household
- Bedrooms per room
- Population per household  

These features help capture **real-world housing density patterns**.

### 4️⃣ Model Building
Implemented and compared multiple regression models:
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

### 5️⃣ Model Evaluation
- Used **Root Mean Squared Error (RMSE)** for performance evaluation
- Applied **cross-validation** to reduce overfitting
- Selected the best-performing model based on validation results

---

## 🧠 Key Learnings

- Importance of proper data preprocessing in ML pipelines
- Impact of feature engineering on model performance
- Why ensemble models (Random Forest) outperform simple linear models on non-linear data
- How cross-validation provides more reliable performance estimates

---

## 🛠️ Technologies Used

- **Python**
- **NumPy & Pandas** — data manipulation
- **Matplotlib & Seaborn** — data visualization
- **Scikit-Learn** — machine learning models & pipelines

---

## 🚀 Future Improvements

- Hyperparameter tuning using `GridSearchCV`
- Experimenting with advanced models like **Gradient Boosting** or **XGBoost**
- Deploying the model using **Flask** or **FastAPI**
- Adding interactive visualizations

---

## 📌 Status

🚧 **Project is currently under active development**

