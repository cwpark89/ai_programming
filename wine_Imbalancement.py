#%%
# Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error

# Read and clean data
file_path = 'WineQT.csv'
df = pd.read_csv(file_path)
df.dropna(inplace=True)
df.columns = df.columns.str.replace(" ", "_")
cleaned_file_path = 'WineQT_Cleaned.csv'
df.to_csv(cleaned_file_path, index=False)

# Load cleaned data
wine = pd.read_csv(cleaned_file_path)
wine.drop('Id', axis=1, inplace=True) # Remove 'Id' column

# Prepare data for modeling
X = wine.drop(columns=["quality", "volatile_acidity"])
y = wine['quality']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Check the distribution after resampling
print(y_train.value_counts(), end='\n\n')

# Define models to train and evaluate
models = {
    'RandomForestRegressor': RandomForestRegressor(),
    'GradientBoostingRegressor': GradientBoostingRegressor(),
}

# Define parameter grids for hyperparameter tuning
param_grids = {
    'RandomForestRegressor': {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    },
    'GradientBoostingRegressor': {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5]
    }
}

# Perform k-fold cross-validation and hyperparameter tuning
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for name, model in models.items():
    print(f"Training and evaluating {name}...")
    
    grid_search = GridSearchCV(model, param_grids[name], cv=kf, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    # Calculate and print evaluation metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    medae = median_absolute_error(y_test, y_pred)
    print(f"{name} - Best Parameters: {grid_search.best_params_}")
    print(f"{name} - Test Mean Absolute Error: {mae:.4f}")
    print(f"{name} - Test Mean Squared Error: {mse:.4f}")
    print(f"{name} - Test Median Absolute Error: {medae:.4f}")
    print(f"Score on training set: {best_model.score(X_train, y_train):.4f}")
    print(f"Score on test set: {best_model.score(X_test, y_test):.4f}")

#%%
# Resample dataset
from imblearn.over_sampling import SMOTE
oversample = SMOTE()
X_resampled, y_resampled = oversample.fit_resample(X, y)

# Split resampled data into train and test sets
X_train_resampled, X_test_resampled, y_train_resampled, y_test_resampled = (
    train_test_split(X_resampled, y_resampled, test_size=0.30, random_state=42)
)

# Check the distribution after resampling
print(y_train_resampled.value_counts())

#%%
# Train and evaluate models on resampled data
for name, model in models.items():
    print(f"Training and evaluating {name} (resampled data)...")
    
    grid_search = GridSearchCV(model, param_grids[name], cv=kf, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train_resampled, y_train_resampled)
    best_model = grid_search.best_estimator_
    y_pred_resampled = best_model.predict(X_test_resampled)
    
    
    # Calculate and print evaluation metrics for resampled data
    mae_resampled = mean_absolute_error(y_test_resampled, y_pred_resampled)
    mse_resampled = mean_squared_error(y_test_resampled, y_pred_resampled)
    medae_resampled = median_absolute_error(y_test_resampled, y_pred_resampled)
    print(f"{name} - Best Parameters: {grid_search.best_params_}")
    print(f"{name} - Test Mean Absolute Error (resampled): {mae_resampled:.4f}")
    print(f"{name} - Test Mean Squared Error (resampled): {mse_resampled:.4f}")
    print(f"{name} - Test Median Absolute Error (resampled): {medae_resampled:.4f}")
    print(f"Score on training set (resampled): {best_model.score(X_train_resampled, y_train_resampled):.4f}")
    print(f"Score on test set (resampled): {best_model.score(X_test_resampled, y_test_resampled):.4f}")

