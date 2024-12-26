# Diabetes Risk Prediction Model

This repository contains a machine learning project that predicts the risk of diabetes based on a dataset of health-related attributes. The project includes data preprocessing, model evaluation, and the selection of the best-performing machine learning model for prediction.

## Project Overview

The goal of this project is to develop a machine learning model that can predict the likelihood of an individual developing diabetes based on their health data. The dataset contains various features such as age, BMI, and blood pressure, which are used to predict a target variable representing the risk score.

### Key Steps:
1. **Data Loading and Preprocessing**: The data is loaded and preprocessed to handle missing values, encode categorical variables, and normalize numerical features.
2. **Model Training and Evaluation**: Several machine learning models are trained and evaluated based on metrics such as Adjusted R-Squared, R-Squared, RMSE (Root Mean Squared Error), and execution time.
3. **Model Selection**: The best-performing models are selected based on their ability to predict diabetes risk accurately and efficiently.

## Models Used

The following machine learning models were evaluated:

- HistGradientBoostingRegressor
- GradientBoostingRegressor
- LGBMRegressor
- XGBRegressor
- RandomForestRegressor
- ExtraTreesRegressor
- AdaBoostRegressor
- MLPRegressor
- DecisionTreeRegressor
- Support Vector Regressor (SVR)
- LinearRegression and others

## Evaluation Metrics

The models were evaluated using the following metrics:

- **Adjusted R-Squared**: Measures the proportion of variance explained by the model, adjusted for the number of predictors.
- **R-Squared**: Represents the proportion of the variance explained by the model.
- **RMSE (Root Mean Squared Error)**: A measure of the model’s prediction error.
- **Time Taken**: The time it took to train the model.

## Best Model: HistGradientBoostingRegressor

After evaluating all models, the **HistGradientBoostingRegressor** was found to be the best choice due to its high performance and efficiency.

- **Adjusted R-Squared**: 0.98
- **R-Squared**: 0.98
- **RMSE**: 2.02
- **Time Taken**: 0.22 seconds

This model outperforms the others in terms of both prediction accuracy and computational efficiency, making it the ideal choice for real-time predictions.


## Dependencies
The following libraries are used in this project:

pandas for data manipulation
numpy for numerical computations
scikit-learn for machine learning models and evaluation
matplotlib and seaborn for data visualization
You can install the required dependencies by running:
  ```bash
  pip install -r requirements.txt
```


## Installation

To run this project locally, follow these steps:


1. Clone the repository:
   ```bash
   git clone https://github.com/mell0wO/Diabete.git
   cd Diabetes

2.Install the required dependencies:

  ```bash
  pip install -r requirements.txt
  ```

## Conclusion
This project demonstrates the use of machine learning to predict diabetes risk based on health-related features. By preprocessing the data and evaluating various models, the HistGradientBoostingRegressor was selected as the most efficient and accurate model for predicting diabetes risk.



