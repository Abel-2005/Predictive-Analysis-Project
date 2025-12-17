Weather Prediction Using Machine Learning
Overview

This project demonstrates the application of predictive analytics and machine learning techniques to real-world weather data. The objective is to analyze atmospheric and air-quality parameters and build models for both regression and classification tasks.

The project focuses on predicting temperature values and classifying weather conditions using supervised learning models, followed by systematic evaluation and comparison to identify the best-performing approaches.

Problem Statement

Weather data is complex and highly non-linear. Traditional linear models often fail to capture these interactions effectively. This project aims to:

Predict temperature using regression techniques

Classify weather conditions using classification algorithms

Compare baseline and advanced models to determine optimal performance

Dataset

Type: Global weather and air-quality data

Size: ~135,000 records, 35 attributes

Source: Originally generated using WeatherAPI and made publicly available via Kaggle

Features include:

Temperature

Humidity

Wind speed

Atmospheric pressure

Precipitation

Air quality indicators (PM2.5, PM10, CO, NO₂)

Note: The dataset was accessed via a GitHub repository, but its original source is WeatherAPI and Kaggle.

Data Preprocessing

The following preprocessing steps were performed:

Missing value handling using median (numerical) and mode (categorical) imputation

Domain-based outlier handling for temperature values

Feature scaling using standardization

Class balancing for the classification task

Feature selection based on data type and relevance

Exploratory Data Analysis

Exploratory analysis was conducted to understand data distribution and relationships, including:

Temperature distribution analysis

Correlation analysis between numerical features

Visual analysis of model performance and predictions

Models Implemented
Regression Models

Linear Regression (baseline)

Random Forest Regressor

Classification Models

Logistic Regression (baseline)

Decision Tree Classifier

Model Evaluation
Regression Metrics

Mean Absolute Error (MAE)

Root Mean Squared Error (RMSE)

R² Score

Classification Metrics

Accuracy

Precision

Recall

F1-score

Confusion Matrix

Visual comparisons were used alongside numerical metrics for comprehensive evaluation.

Results

Best Regression Model: Random Forest Regressor

Best Classification Model: Decision Tree Classifier

Tree-based models outperformed linear models due to their ability to capture non-linear relationships in weather data.

Technologies Used

Python

Pandas

NumPy

Scikit-learn

Matplotlib

Seaborn

Project Structure
├── data/
│   └── weather.csv
├── notebooks/
│   └── weather_analysis.ipynb
├── src/
│   └── model_training.py
├── README.md

Conclusion

This project highlights the effectiveness of machine learning techniques in weather prediction tasks. Ensemble and tree-based models demonstrated superior performance over linear models, validating their suitability for complex environmental data.

Future Enhancements

Time-series forecasting using LSTM or ARIMA

Integration of real-time weather APIs

Deployment as a web application

Advanced feature engineering and deep learning models

References

Kaggle: Global Weather Repository

WeatherAPI.com

Scikit-learn Documentation

Author

Abel B Varughese
