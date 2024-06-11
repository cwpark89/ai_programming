#%%

import pandas as pd
import numpy as np
import pycaret
import sys
import os 
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from matplotlib.ticker import FormatStrFormatter
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression, LogisticRegressionCV, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

plt.style.use('ggplot')

# Define the path to the CSV file
file_path = '/home/aiprogramming/Desktop/wine/WineQT.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Remove rows with any column having NA/null data
df_cleaned = df.dropna()
#df_cleaned = df.drop('Id', axis=1)

# You might want to reset index if you plan to export this cleaned dataset
df_cleaned.reset_index(drop=True, inplace=True)

# Change blank to underbar
df_cleaned.columns = df.columns.str.replace(" ","_")

# Optionally, save the cleaned DataFrame back to a new CSV file
cleaned_file_path = '/home/aiprogramming/Desktop/wine/WineQT_Cleaned.csv'
df_cleaned.to_csv(cleaned_file_path, index=False)


print("Null data removed and clean CSV saved.")

%matplotlib inline
plt.style.use('ggplot')

#%%
wine = pd.read_csv('/home/aiprogramming/Desktop/wine/WineQT_Cleaned.csv')
wine.shape

wine = df.drop('Id',axis=1)
wine.shape

#%%
wine.info()
# %%
wine.head()
# %%
wine.groupby('quality').mean()
# %%
wine.hist(figsize=(12, 10))
plt.suptitle('Histogram of Each Numeric Column')
plt.show()

#%%
# num_features = wine.drop('quality', axis=1).shape[1]  # Number of features excluding 'quality'
# features = wine.drop('quality', axis=1).columns      # Feature names excluding 'quality'

# figure, axs = plt.subplots(nrows=6, ncols=2, figsize=(20, 30))  # Adjust the figure size as needed
# figure.suptitle('Bar Plots of Features vs Quality')

# # Loop through the features and create bar plots
# for i, feature in enumerate(features):
#     row = i // 2  # Integer division to find row index
#     col = i % 2   # Remainder to find column index
#     sns.barplot(data=wine, x=feature, y='quality', ax=axs[row, col])
    
#     axs[row, col].tick_params(axis='x', rotation=90, labelright=True) # Rotate 90 deg of x values

# plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
# plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust to prevent suptitle overlap

# plt.show()

#%%
wine.hist(figsize=(12, 10))
plt.suptitle('Histogram of Each Numeric Column')
plt.show()

# %%
corr = wine.corr()
# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

plt.figure(figsize=(12, 10))
sns.heatmap(corr, mask = mask, annot=True, cmap='YlOrRd')
plt.title('Correlation Matrix')
plt.show()

# %%
figure, axes = plt.subplots(ncols=2, nrows=2)
figure.set_size_inches(12,10)

sns.displot(wine["quality"], ax=axes[0][0])
stats.probplot(wine["quality"], dist='norm', fit=True, plot=axes[0][1])
sns.displot(np.log(wine["quality"]), ax=axes[1][0])
stats.probplot(np.log1p(wine["quality"]), dist='norm', fit=True, plot=axes[1][1])


# %%
X = wine.drop(columns="quality")
X = X.drop(columns="volatile acidity")
y = df['quality']

#%%
# Normalizing the features
# scaler = StandardScaler()
# X_normalized = scaler.fit_transform(X)

#%%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

print("X Train : ", X_train.shape)
print("X Test  : ", X_test.shape)
print("Y Train : ", y_train.shape)
print("Y Test  : ", y_test.shape)

# %%
# using the model LinearRegression
LR_model=LinearRegression()

# fit model
LR_model.fit(X_train,y_train)

# Score X and Y - test and train
print( " **************** LinearRegression **************** ")
print("Score the X-train with Y-train is : ", LR_model.score(X_train,y_train))
print("Score the X-test  with Y-test  is : ", LR_model.score(X_test,y_test))

# Expected value Y using X test
y_pred_LR=LR_model.predict(X_test)

# Model Evaluation
print( " Model Evaluation Linear R : mean absolute error is ", mean_absolute_error(y_test,y_pred_LR))
print(" Model Evaluation Linear R : mean squared  error is " , mean_squared_error(y_test,y_pred_LR))
print(" Model Evaluation Linear R : median absolute error is " ,median_absolute_error(y_test,y_pred_LR)) 
print(" Model Evaluation Linear R : r-squired score is " ,r2_score(y_test,y_pred_LR))

# %%
# using the model RandomForestRegressor
LR_model=RandomForestRegressor()

# fit model
LR_model.fit(X_train,y_train)

# Score X and Y - test and train
print( " **************** RandomForestRegressor **************** ")
print("Score the X-train with Y-train is : ", LR_model.score(X_train,y_train))
print("Score the X-test  with Y-test  is : ", LR_model.score(X_test,y_test))

# Expected value Y using X test
y_pred_LR=LR_model.predict(X_test)

# Model Evaluation
print( " Model Evaluation Linear R : mean absolute error is ", mean_absolute_error(y_test,y_pred_LR))
print(" Model Evaluation Linear R : mean squared  error is " , mean_squared_error(y_test,y_pred_LR))
print(" Model Evaluation Linear R : median absolute error is " ,median_absolute_error(y_test,y_pred_LR)) 
print(" Model Evaluation Linear R : r-squired score is " ,r2_score(y_test,y_pred_LR))
# %%
models = {
    'LinearRegression': LinearRegression(),
    'LogisticRegression': LogisticRegression(),
    'LogisticRegressionCV': LogisticRegressionCV(),
    'Lasso': Lasso(),
    'RandomForestRegressor': RandomForestRegressor(),
    'GradientBoostingRegressor': GradientBoostingRegressor(),
    'SVR': SVR(),
    'KNeighborsRegressor': KNeighborsRegressor(),
    'DecisionTreeRegressor': DecisionTreeRegressor()
}
# models = {
#     'RandomForestRegressor': RandomForestRegressor(),
#     'GradientBoostingRegressor': GradientBoostingRegressor(),
# }

for name, model in models.items():
    print(f"Training and evaluating {name}...")
    model.fit(X_train, y_train)
    y_pred_LR = model.predict(X_test)
    mae = mean_absolute_error(y_test,y_pred_LR)
    mse = mean_squared_error(y_test, y_pred_LR)
    medae = median_absolute_error(y_test,y_pred_LR)
    r2 = r2_score(y_test, y_pred_LR)
    print(f"{name}, - Mean Absolute Error: {mae}, Mean Squared Error: {mse}, Median Absolute Error: {medae}, R-squared: {r2}")
    print("Score the X-train with Y-train is : ", model.score(X_train,y_train))
    print("Score the X-test  with Y-test  is : ", model.score(X_test,y_test))
# %%
