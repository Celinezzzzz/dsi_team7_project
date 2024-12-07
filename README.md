# **Stock Index Price Direction Prediction using Machine Learning**

## **Purpose & Overview**

This project aims to predict the next-day direction of stock index prices (whether the price will increase or decrease) using historical stock market data and machine learning techniques. We focus on various stock market indicators such as volume, volatility, open, adjust close, high and low prices, along with technical indicators like MACD, RSI, EMA and SMA to build our predictive model.

The objective of the project is to develop and compare different machine learning models to determine which one performs best in predicting stock index price trends, helping traders and investors make data-driven decisions.

## **Goals & Objectives**

The primary goals of this project are:

- **Objective**: To accurately predict the next-day closing stock index price direction using historical stock data and machine learning algorithms.
- **Success Criteria**:
  - Achieving a high prediction accuracy score (target > 70%) for predicting the direction of stock price movements.
  - Employing effective feature selection and preprocessing techniques to handle missing data and outliers.
  - Comparing multiple machine learning algorithms to determine the best-performing model.
  - Validating models using metrics like accuracy, ROC and AUC for robust evaluation.

## **Techniques & Technologies**

The following tools and methods were used to complete the project:

### **Techniques**:

1. **Data Preprocessing**:

   - Handling missing values and outliers by capping feature values at the 1st and 99th percentiles and leveraging forward and backward filling.
   - Removing duplicates and filtering rows based on non-null values for essential columns.
   - Creating new features such as moving averages (5-day, 10-day, 30-day), volatility, and percentage returns.

2. **Feature Engineering**:

   - Technical indicators like Moving Average Convergence Divergence (MACD), Relative Strength Index (RSI), Exponential Moving Average (EMA), and Simple Moving Average (SMA) were added to enhance model input.
   - The target variable, price direction (1 for increase, 0 for decrease or no change), was created to be used in classification models.

3. **Machine Learning Models**:

   - **Logistic Regression**: A simple, interpretable model used as a baseline.
   - **Random Forest Classifier**: An ensemble learning method that performs well with non-linear relationships and handles feature importance effectively.

4. **Model Evaluation**:

   - Evaluation metrics such as Accuracy, ROC AUC Score, and Classification Report (Precision, Recall, F1 Score) were used to assess model performance.
   - ROC curves were plotted for both models, enabling visual comparison of their true positive and false positive rates.

5. **Exploratory Data Analysis (EDA)**:
   - Dataset overview, shape, time period, and number of indexes included.
   - Calculation of summary statistics (mean, min, max, standard deviation) for key features.
   - Visualizations of feature correlation and time series trend analysis for stock indicators.

### **Technologies**:

- **Python**: Programming language used for data manipulation and machine learning.
- **Pandas**: For data preprocessing and feature engineering.
- **SQL**: Used through `pandasql` to query and filter the dataset.
- **Scikit-learn**: For building and evaluating machine learning models.
- **StockStats**: For calculating advanced stock indicators such as MACD, BBP, RSI, EMA and SMA.
- **Seaborn & Matplotlib**: For creating visualizations such as correlation heatmaps, trend analysis, and feature importance plots.

## **Visualization**

Visualizations played a key role in the analysis of the dataset and the model’s results. The following visualizations were created:

1. **Trend Analysis**: Line graphs displaying stock index price movements over time, with moving averages for trend analysis on individual stock indexes.

2. **Feature Correlation Matrix**: A heatmap visualizing the correlation between various features, helping with feature selection and identifying relationships between indicators.

3. **Feature Importance Plot**: Visualizes the most significant features contributing to the Random Forest model’s predictions, using bar plots to display feature importances.

## **Key Findings & Instructions**

### **Key Findings**:

- **Logistic Regression** performed slightly better than Random Forest, with higher ROC AUC and accuracy scores.
- Features such as **rsi_12**,**volatility**, **volume**, and **MACD** were found to be significant predictors of the next day's stock price direction.
- **Trend Analysis** plots indicate that price movements and certain indicators align over time, providing a visual cue for model feature selection.

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

4. **Exploratory Data Analysis:** Obtain insights and trends of the processed dataset by running **exploratory_data_analysis.ipynb**

5. **Train Models:** Execute the classification script to train both Logistic Regression and Random Forest models by running **classification.ipynb**

6. **View Results:** After the models are trained, results will be printed in the terminal.

## **Conclusion**

In conclusion, the project successfully demonstrated the predictive capabilities of machine learning models in stock market analysis. The Logistic Regression model proved to be the best-performing model for this problem, with a reliable accuracy and high ROC AUC score, indicating good predictive performance on stock index direction.

For future work, we recommend:

- Testing additional advanced models such as XGBoost or LSTM for time-series data, which may improve performance on more complex stock data patterns.
- Incorporating more features such as news sentiment or macroeconomic indicators to enhance predictions.
- Implementing GridSearchCV or RandomizedSearchCV for comprehensive hyperparameter tuning to optimize model performance. Fine-tuning parameters such as the number of estimators, maximum depth, and minimum samples split in Random Forest, for example, can improve model accuracy and generalizability.

## **Credits**

This project was developed collaboratively by the following team members:

- Besher Mahrouka:

  1- Built and prepared the GitHub repo for the project

  2- Built the data cleaning code as part of data engineering and feature engineering

  3- Built classification code which includes liner regression and random forest

  4- Built a deep learning model (LSTM) to predict stock market

  [Video_recording_Besher_Mahrouka](https://drive.google.com/file/d/1GfPUTptkUZ9XtsS35yL9JuxYm-J6R23h/view?usp=drive_link)


- Si Min Zhou:

  1- Enhanced data cleaning and classification code, refining indicator selection to improve model accuracy; added evaluation methods to better assess model performance

  2- Designed and implemented Exploratory Data Analysis code for in-depth visualization, dataset analysis, and feature selection

  3- Authored the README documentation, ensuring clear and thorough project communication

- Shweta Sharma:
- Anna Hromova:
1- Developed the SQL code to upload CSV data into the database.

2- Ensured the proper formatting and structure of data before inserting it into the SQL tables for further analysis.

3- Assisted with the integration of SQL-based data management into the overall project workflow.
