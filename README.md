# **Stock Index Price Direction Prediction using Machine Learning and Deep Learning**

## **Purpose & Overview**

This project aims to predict the next-day direction of stock index prices (whether the price will increase or decrease) using historical stock market data and machine learning techniques. We focus on various stock market indicators such as volume, volatility, open, adjust close, high and low prices, along with technical indicators like MACD, RSI, EMA and SMA to build our predictive models.

The objective of the project is to develop and compare different machine learning models to determine which one performs best in predicting stock index price trends, helping traders and investors make data-driven decisions.

## **Goals & Objectives**

The primary goals of this project are:

- **Objective**: To accurately predict the next-day closing stock index price direction using historical stock data and machine learning algorithms.
- **Success Criteria**:
  - Achieving a high prediction accuracy score (target > 70%) for predicting the direction of stock price movements.
  - Employing effective feature selection and preprocessing techniques to handle missing data and outliers.
  - Comparing machine learning models (Logistic Regression, Random Forest) with deep learning models (LSTM).
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

4. **Deep Learning Model**:

   - Implemented an **LSTM (Long Short-Term Memory)** network to leverage sequential data for prediction:
     - Added layers for sequence modeling, dropout layers for regularization, and an attention mechanism to focus on relevant data patterns.
     - Applied techniques like hyperparameter tuning (batch size, learning rate, and sequence length) to improve performance.

5. **Model Evaluation**:

   - Evaluation metrics such as Accuracy, ROC AUC Score, and Classification Report (Precision, Recall, F1 Score) were used to assess model performance.
   - ROC curves were plotted for machine learning models, and training history (accuracy/loss) was visualized for deep learning models.

6. **Exploratory Data Analysis (EDA)**:

   - Dataset overview, shape, time period, and number of indexes included.
   - Calculation of summary statistics (mean, min, max, standard deviation) for key features.
   - Visualizations of feature correlation and time series trend analysis for stock indicators.

### **Technologies**:

- **Python**: Programming language used for data manipulation and machine learning.
- **Pandas**: For data preprocessing and feature engineering.
- **SQL**: Used through `pandasql` to query and filter the dataset.
- **Scikit-learn**: For building and evaluating machine learning models.
- **Keras**: For designing and training deep learning models.
- **StockStats**: For calculating advanced stock indicators such as MACD, BBP, RSI, EMA, and SMA.
- **Seaborn & Matplotlib**: For creating visualizations such as correlation heatmaps, trend analysis, and feature importance plots.

## **Visualization**

Visualizations played a key role in the analysis of the dataset and the model’s results. The following visualizations were created:

1. **Trend Analysis**: Line graphs displaying stock index price movements over time, with moving averages for trend analysis on individual stock indexes.

2. **Feature Correlation Matrix**: A heatmap visualizing the correlation between various features, helping with feature selection and identifying relationships between indicators.

3. **Feature Importance Plot**: Visualizes the most significant features contributing to the Random Forest model’s predictions, using bar plots to display feature importances.

4. **Deep Learning Training History**: Plots of accuracy and loss for training and validation sets, showing model optimization over epochs.

## **Key Findings & Instructions**

### **Key Findings**:

- **Logistic Regression** performed slightly better than Random Forest, with higher ROC AUC and accuracy scores.
- The **LSTM model** demonstrated potential in capturing sequential patterns but still requires further optimization to surpass machine learning models.
- Features such as **rsi_12**, **volatility**, **volume**, and **MACD** were found to be significant predictors of the next day's stock price direction.
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

6. **Train Deep Learning Model:** Execute to train and evaluate the LSTM model by running **deep_learning.ipynb**

7. **View Results:** After the models are trained, results will be printed in the terminal.

## **Conclusion**

In conclusion, the project successfully demonstrated the predictive capabilities of both machine learning and deep learning models in stock market analysis. Logistic Regression emerged as the most reliable model in this iteration, achieving the best balance of accuracy and interpretability. The LSTM model showcased promising potential for sequential data modeling but requires further tuning to enhance its accuracy.

For future work, we recommend:

- Testing additional advanced models such as XGBoost or other deep learning architectures.
- Incorporating more features such as news sentiment or macroeconomic indicators to enhance predictions.
- Implementing GridSearchCV or RandomizedSearchCV for hyperparameter optimization in machine learning models.
- Exploring more extensive tuning of LSTM parameters, including sequence length, hidden layers, and attention mechanisms.

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

  3- Optimized the LSTM deep learning model by fine-tuning parameters and improving model architecture for better performance

  4- Authored the README documentation, ensuring clear and thorough project communication

  [Video_recording_Si_Min_Zhou](https://drive.google.com/file/d/1xF9PDBUXlvQyntUiN7AMyb39-ukTWPlp/view?usp=sharing)

- Shweta Sharma:
- Anna Hromova:

  1- Developed the SQL code to upload CSV data into the database.

  2- Ensured the proper formatting and structure of data before inserting it into the SQL tables for further analysis.

  3- Assisted with the integration of SQL-based data management into the overall project workflow.

  [Recording_Anna_Hromova](https://drive.google.com/file/d/1lMqlNGspNQBC6xDBbuEjxEtxgG7dJyaZ/view?usp=drivesdk)
