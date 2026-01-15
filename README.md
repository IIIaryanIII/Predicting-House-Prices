# Predicting-House-Prices (Still in progress, last updated - 15/01/2026)
This project focuses on building an end-to-end machine learning regression pipeline to predict median house prices in California districts using socio-economic and geographic features derived from the 1990 California census dataset.
📊 Dataset Description

The dataset contains information about various housing districts in California. Each row represents aggregated data for a district rather than individual houses.

Key Features

Geographical attributes: longitude, latitude

Housing attributes: housing median age, total rooms, total bedrooms

Demographic attributes: population, households

Economic attribute: median income

Categorical attribute: ocean proximity

Target Variable

Median House Value – the value to be predicted

⚙️ Project Workflow

Exploratory Data Analysis (EDA)

Visualized geographical price trends using latitude & longitude

Analyzed feature distributions and correlations

Identified median income as a strong predictor of house prices

Data Preprocessing

Handled missing values using imputation techniques

Performed feature scaling for numerical attributes

Encoded categorical features using one-hot encoding

Feature Engineering

Created derived features such as:

Rooms per household

Bedrooms per room

Population per household

These features helped capture real-world housing density patterns

Model Building
Implemented and compared multiple regression models:

Linear Regression

Decision Tree Regressor

Random Forest Regressor

Model Evaluation

Used Root Mean Squared Error (RMSE) for evaluation

Applied cross-validation to reduce overfitting

Selected the best performing model based on validation performance

🧠 Key Learnings

Importance of proper data preprocessing in ML pipelines

Impact of feature engineering on model performance

Why ensemble models (Random Forest) outperform simple linear models for non-linear data

How cross-validation provides more reliable performance estimates

🛠️ Technologies Used

Python

NumPy & Pandas – data manipulation

Matplotlib & Seaborn – visualization

Scikit-Learn – machine learning models and pipelines

🚀 Future Improvements

Hyperparameter tuning using GridSearchCV

Trying advanced models like Gradient Boosting or XGBoost

Deploying the model using Flask or FastAPI

Adding interactive visualizations
