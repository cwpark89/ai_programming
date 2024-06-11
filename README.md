
# Wine Quality Prediction - README

## Overview
This repository contains a Python script designed to predict wine quality based on various physicochemical properties. The script performs data preprocessing, trains multiple regression models, and evaluates their performance, both on the original and resampled datasets to address class imbalance.

## Features
- Data cleaning and preprocessing
- Training and evaluation of `RandomForestRegressor` and `GradientBoostingRegressor`
- Hyperparameter tuning using GridSearchCV
- Handling class imbalance with SMOTE (Synthetic Minority Over-sampling Technique)
- Evaluation of model performance using various metrics

## Requirements
To run the script, you need to have the following packages installed:
- numpy==1.24.2
- pandas==1.4.4
- scipy==1.10.1
- matplotlib==3.6.0
- matplotlib-inline==0.1.7
- seaborn==0.13.2
- scikit-learn==1.1.1
- pycaret==3.2.0

You can install the necessary packages using pip:
```bash
pip install -r requirements.txt
```

## Files
- `WineQT.csv`: The original dataset containing wine quality data.
- `WineQT_Cleaned.csv`: The cleaned dataset after preprocessing.
- `wine_Imbalancement.py`: The main script for data preprocessing, model training, evaluation, and handling class imbalance.

## Usage
1. **Data Cleaning and Preprocessing**
   - The script reads the dataset (`WineQT.csv`), removes any missing values, and cleans column names.
   - The cleaned data is saved to `WineQT_Cleaned.csv`.

2. **Model Training and Evaluation**
   - The script defines two regression models: `RandomForestRegressor` and `GradientBoostingRegressor`.
   - It performs hyperparameter tuning using `GridSearchCV` with a 5-fold cross-validation.
   - The models are trained and evaluated on the original dataset, with performance metrics printed for each model.

3. **Handling Class Imbalance**
   - The script uses SMOTE to resample the dataset to address class imbalance.
   - The resampled data is split into training and test sets.
   - The models are re-trained and evaluated on the resampled data, with performance metrics printed for each model.

## Script Breakdown
The script is divided into the following sections:

1. **Imports**
   - Necessary libraries for data manipulation, model training, and evaluation are imported.

2. **Data Cleaning**
   - Reads the dataset, removes missing values, and cleans column names.

3. **Data Preparation**
   - Prepares the data for modeling by splitting it into features (`X`) and target (`y`).
   - Splits the data into training and test sets.

4. **Model Training and Evaluation**
   - Defines the models and parameter grids for hyperparameter tuning.
   - Performs k-fold cross-validation and hyperparameter tuning.
   - Trains and evaluates the models, printing the evaluation metrics.

5. **Handling Class Imbalance**
   - Resamples the dataset using SMOTE.
   - Splits the resampled data into training and test sets.
   - Re-trains and evaluates the models on the resampled data, printing the evaluation metrics.

## Example Output
The script prints the best parameters found during grid search, along with the evaluation metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), and Median Absolute Error (MedAE) for both the original and resampled datasets.

## Contributions
Contributions to this repository are welcome. Feel free to submit a pull request or open an issue for any enhancements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

---

For any questions or further information, please contact Changwoo Park at cwpark89@soongsil.ac.kr.
