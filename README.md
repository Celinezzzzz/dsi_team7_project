# **Stock Index Price Direction Prediction using Machine Learning**

## **Purpose & Overview**

This project aims to predict the next-day direction of stock index prices (whether the price will increase or decrease) using historical stock market data and machine learning techniques. We focus on various stock market indicators such as volume, open, adjust close, high and low prices, along with technical indicators like MACD, BBP, RSI, EMA and SMA to build our predictive model.

The objective of the project is to develop and compare different machine learning models to determine which one performs best in predicting stock index price trends, helping traders and investors make data-driven decisions.

## **Goals & Objectives**

The primary goals of this project are:

- **Objective**: To accurately predict the next-day closing stock index price direction using historical stock data and machine learning algorithms.
- **Success Criteria**:
  - A high prediction accuracy score (> 70%) for predicting the direction of stock price movements.
  - Effective feature selection and preprocessing techniques to handle missing data and outliers.
  - Comparison of multiple machine learning algorithms to determine the best-performing model.
  - Proper validation of models using cross-validation and hyperparameter tuning techniques?

## **Techniques & Technologies**

The following tools and methods were used to complete the project:

### **Techniques**:

1. **Data Preprocessing**:

   - Handling missing values and outliers by capping feature values at the 1st and 99th percentiles.
   - Creating new features such as moving averages (5-day, 10-day, 30-day), volatility, and percentage returns.

2. **Feature Engineering**:

   - Technical indicators like Moving Average Convergence Divergence (MACD), Bollinger Band Percentage(BBP), Relative Strength Index (RSI), SMA and EMA were used to improve the predictive power of the models.
   - Price direction (`1` for increase, `0` for decrease or no change) was used as the target variable.

3. **Machine Learning Models**:

   - **Logistic Regression**: A simple, interpretable model used as a baseline.
   - **Random Forest Classifier**: An ensemble learning method that performs well with non-linear relationships and handles feature importance effectively.

4. **Model Evaluation**:
   - Metrics such as **Accuracy** and **Classification Report** (Precision, Recall, F1 Score) were used to evaluate the models?
   - Cross-validation and hyperparameter tuning using **GridSearchCV** to optimize model performance?

### **Technologies**:

- **Python**: Programming language used for data manipulation and machine learning.
- **Pandas**: For data preprocessing and feature engineering.
- **SQL**: Used through `pandasql` to query and filter the dataset.
- **Scikit-learn**: For building and evaluating machine learning models.
- **StockStats**: For calculating advanced stock indicators such as MACD, BBP, RSI, EMA and SMA.

## **Visualization**

Visualizations played a key role in the analysis of the dataset and the model’s results. The following visualizations were created:

1. **Trend Analysis**: A line graph displaying stock index price movements over time with moving averages for trend analysis?

2. **Feature Importance Plot**: Visualizes the most significant features contributing to the Machine Learning models’ predictions?

## **Key Findings & Instructions**

### **Key Findings**:

- The **A model** performed better than B model, achieving a higher accuracy of `X%` compared to Logistic Regression's `Y%`?
- Features like **A**, **B**, and **C** were found to be significant predictors of the next day’s stock price direction.
- **Cross-validation** showed xxx?

### **Instructions for Setup**:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/beshermahrouka/dsi_team7_project.git
   ```
2. **Install Dependencies:** Make sure you have Python installed and use pip or conda to install the necessary libraries:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run Data Preprocessing:** Clean and preprocess the raw dataset by running **data_cleaning.ipynb**

4. **Train Models:** Execute the classification script to train both Logistic Regression and Random Forest models by running **classification.ipynb**

5. **View Results:** After the models are trained, results will be printed in the terminal.

## **Conclusion**

In conclusion, the project successfully demonstrated the predictive capabilities of machine learning models in stock market analysis. The A model proved to be the best-performing model for this problem, with high accuracy and stable performance across different validation sets.

For future work, we recommend:

- Testing more advanced models such as XGBoost or LSTM for time-series data.
- Incorporating more features such as news sentiment or macroeconomic indicators to enhance predictions.

## **Credits**

This project was developed collaboratively by the following team members:

- Besher Mahrouka:
- Si Min Zhou:
- Shweta Sharma:
- Anna Hromova:
