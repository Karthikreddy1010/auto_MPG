# Importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import collections
import math
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV




# Task 1 Data Loading

# Define column names
column_names = ["mpg", "cylinders", "displacement", "horsepower", "weight",
                "acceleration", "model_year", "origin", "car_name"]
rawData = pd.read_csv("auto-mpg.csv", names=column_names, header=None, sep=r'\s+')

# Compute number of rows and columns
dataRows, dataCols = rawData.shape
print(f"Number of Rows: {dataRows}")
print(f"Number of Columns: {dataCols}")

# Print the last 10 rows
print(rawData.tail(10))

# Task 2 Data Preprocessing

# Checking null values in a rawData
missing_values=rawData.isnull().sum()
print("Missing values in each column:\n",missing_values)
# convert horsepower column to numeric
rawData['horsepower']=pd.to_numeric(rawData["horsepower"],errors="coerce")
# Fill missing values with the median for specific columns
for col in ['horsepower', 'mpg']:
    rawData[col] = rawData[col].fillna(rawData[col].median())
print(rawData[['horsepower', 'mpg']].isna().sum())
# store the processed data in a preprocesseddata
preprocessedData = rawData.copy()
print("Summary after preprocessing:\n",preprocessedData.describe())
#### Task 3 Data Analysis
averageMPG=preprocessedData['mpg'].mean()
print(f"Average MPG:{averageMPG}")

# Extracting vehicle types from 'car name' (assuming last word in name hints the type)
vehicle_types = [name.split()[-1] for name in preprocessedData["car_name"]]
commonVehicleType = collections.Counter(vehicle_types).most_common(1)[0][0]

print(f"Most Common Vehicle Type: {commonVehicleType}")
# Most frequently occurring cyclinder count
commonCylinderCount = preprocessedData["cylinders"].mode()[0]
print(f"Most Common Cylinder Count: {commonCylinderCount}")

def standardDeviation(arr):
    mean=sum(arr)/len(arr)
    variance=sum((x-mean)**2 for x in arr)/len(arr)
    return math.sqrt(variance)
print(f"Standard Deviation of MPG: {standardDeviation(preprocessedData['mpg'])}")
# correlation coefficient for columns relationship

def correlationCoefficient(x, y):
    mean_x, mean_y = sum(x) / len(x), sum(y) / len(y)
    numerator = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y))
    denominator = math.sqrt(sum((a - mean_x) ** 2 for a in x) * sum((b - mean_y) ** 2 for b in y))
    return numerator / denominator

# Example usage:
print(f"Correlation between MPG and Weight: {correlationCoefficient(preprocessedData['mpg'], preprocessedData['weight'])}")
# correlation table for MPG, Horsepower and Weight
attributeCorrelations = preprocessedData[["mpg", "horsepower", "weight"]].corr()
print(attributeCorrelations)
"""Notable Correlations
A negative correlation between MPG and weight suggests that heavier cars tend to have lower fuel efficiency.
A negative correlation between MPG and horsepower suggests that more powerful engines tend to consume more fuel"""
# Compute correlation betwen engine Displacement and MPG
correlationDispMPG = rawData["mpg"].corr(preprocessedData["displacement"])
print(f"Correlation Between Engine Displacement and MPG: {correlationDispMPG}")
# To identify the most influential attribute for MPG
# Selecting only numeric columns
numericData = preprocessedData.select_dtypes(include=['number'])
# Compute correlations with MPG
mpg_correlations = numericData.corr()["mpg"].abs().drop("mpg")

# Find the most influential attribute
mostInfluentialAttribute = mpg_correlations.idxmax()
print(f"Most Important Attribute Influencing MPG: {mostInfluentialAttribute}")

# Task 4 Feature Engineering

# vehicle age
current_year=2025
preprocessedData['vehicle_age'] = current_year - preprocessedData['model_year']
# Normalization Min Max Scaling technique

from sklearn.preprocessing import MinMaxScaler
# Normalization only for numeric columns so, selecting only numeric columns
numerical_cols=preprocessedData.select_dtypes(include=['number']).columns

# applying MinMaxScaler to numerical columns
scaler=MinMaxScaler()
preprocessedData[numerical_cols]=scaler.fit_transform(preprocessedData[numerical_cols])
# Extract the make from Car Name

preprocessedData['make']=preprocessedData['car_name'].apply(lambda x:x.split()[0])

# encoding label to convert categorical columns into numerical columns
preprocessedData['make_label']=preprocessedData['make'].astype('category').cat.codes
# updated dataset
print(preprocessedData.head())

# Task 5 Data Visualization
# Histogram for the MPG
"""Histogram/KDE Plot: Shows the distribution of MPG in the dataset, 
with a density curve for a smoother visualization."""
sns.set_theme(style="whitegrid")
plt.figure(figsize=(9, 6))
sns.histplot(preprocessedData['mpg'], kde=True, bins=25, color='royalblue', stat='density', linewidth=1)
plt.title('Distribution of MPG', fontsize=16, fontweight='bold')
plt.xlabel('MPG', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.grid(True)
plt.show()

# 2. Bar Plot for the Frequency of Entries for Each Make (Top 10 Most Frequent Makes)
"""Top 10 Makes Bar Plot: Displays the frequency of entries for the top 10 makes in the dataset. 
The color palette helps highlight the frequencies."""
top_makes = preprocessedData['make'].value_counts().head(10)

# Create the bar plot with color palette applied
plt.figure(figsize=(9, 6))
sns.barplot(x=top_makes.index, y=top_makes.values)
plt.title('Top 10 Most Frequent Vehicle Makes', fontsize=16, fontweight='bold')
plt.xlabel('Vehicle Make', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# 3. Box Plot of MPGs for Different Makes (Top 10 Most Frequent Makes)
top_10_makes = top_makes.index

plt.figure(figsize=(9, 7))
sns.boxplot(x='make', y='mpg', data=preprocessedData[preprocessedData['make'].isin(top_10_makes)], hue='make')
plt.title('MPG Distribution by Vehicle Make', fontsize=16, fontweight='bold')
plt.xlabel('Vehicle Make', fontsize=12)
plt.ylabel('MPG', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.grid(True)
plt.show()

# 4. Scatter Plots for Relationships Between Specific Features and MPG

# MPG vs Horsepower
plt.figure(figsize=(9, 6))
sns.scatterplot(x='horsepower', y='mpg', data=preprocessedData, color='darkred', s=100, edgecolor='black', alpha=0.7)
plt.title('MPG vs Horsepower', fontsize=16, fontweight='bold')
plt.xlabel('Horsepower', fontsize=12)
plt.ylabel('MPG', fontsize=12)
plt.grid(True)
plt.show()
# MPG vs Weight
plt.figure(figsize=(9, 6))
sns.scatterplot(x='weight', y='mpg', data=preprocessedData, color='darkgreen', s=100, edgecolor='black', alpha=0.7)
plt.title('MPG vs Weight', fontsize=16, fontweight='bold')
plt.xlabel('Weight', fontsize=12)
plt.ylabel('MPG', fontsize=12)
plt.grid(True)
plt.show()
# MPG vs Displacement
plt.figure(figsize=(9, 6))
sns.scatterplot(x='displacement', y='mpg', data=preprocessedData, color='navy', s=100, edgecolor='black', alpha=0.7)
plt.title('MPG vs Displacement', fontsize=16, fontweight='bold')
plt.xlabel('Displacement', fontsize=12)
plt.ylabel('MPG', fontsize=12)
plt.grid(True)
plt.show()

preprocessedData.head()
# Task 6 Model Building

# taking all numerical features and excluding categorical columns

X=preprocessedData[['cylinders','displacement','horsepower','weight','acceleration',
                   'model_year','origin','vehicle_age','make_label']]
y=preprocessedData['mpg']

# splitting the data for training=70%, validation purpose=15% , and testing =15%

X_train,X_temp,y_train,y_temp=train_test_split(X,y,test_size=0.3,random_state=42)

X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print(f"Training Set: {X_train.shape}, Validation Set: {X_val.shape}, Test Set: {X_test.shape}")


# train the model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# predictions
# Predictions
y_pred_train = rf_model.predict(X_train)
y_pred_test = rf_model.predict(X_test)

# Evaluation Metrics
mae_train = mean_absolute_error(y_train, y_pred_train)
mae_test = mean_absolute_error(y_test, y_pred_test)

mse_train = mean_squared_error(y_train, y_pred_train)
mse_test = mean_squared_error(y_test, y_pred_test)

rmse_train = np.sqrt(mse_train)
rmse_test = np.sqrt(mse_test)

r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)

print(f"Training MAE: {mae_train:.2f}")
print(f"Test MAE: {mae_test:.2f}")

print(f"Training MSE: {mse_train:.2f}")
print(f"Test MSE: {mse_test:.2f}")

print(f"Training RMSE: {rmse_train:.2f}")
print(f"Test RMSE: {rmse_test:.2f}")

print(f"Training R² Score: {r2_train:.2f}")
print(f"Test R² Score: {r2_test:.2f}")

# Task 7 Model Tuning
# Define the parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],  
    'max_depth': [None, 10, 20],      
    'min_samples_split': [2, 5, 10],  
    'min_samples_leaf': [1, 2, 4]     
}

# Initialize the model
rf = RandomForestRegressor(random_state=42)

# Perform Grid Search
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, 
                           cv=5, scoring='r2', n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Best hyperparameters
best_params = grid_search.best_params_
print("Best Parameters:", best_params)

# Train the Random Forest model with the best parameters
rf_optimized = RandomForestRegressor(**best_params, random_state=42)
rf_optimized.fit(X_train, y_train)

# Predictions
y_pred_train = rf_optimized.predict(X_train)
y_pred_test = rf_optimized.predict(X_test)

# Evaluation Metrics
mae_train = mean_absolute_error(y_train, y_pred_train)
mae_test = mean_absolute_error(y_test, y_pred_test)

mse_train = mean_squared_error(y_train, y_pred_train)
mse_test = mean_squared_error(y_test, y_pred_test)

rmse_train = np.sqrt(mse_train)
rmse_test = np.sqrt(mse_test)

r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)

# Print results
print(f"Training MAE: {mae_train:.2f}")
print(f"Test MAE: {mae_test:.2f}")

print(f"Training MSE: {mse_train:.2f}")
print(f"Test MSE: {mse_test:.2f}")

print(f"Training RMSE: {rmse_train:.2f}")
print(f"Test RMSE: {rmse_test:.2f}")

print(f"Training R² Score: {r2_train:.2f}")
print(f"Test R² Score: {r2_test:.2f}")

# Task 8 Deployment

def predictMPG(model, input_features):
    """
    Predicts the miles per gallon (MPG) based on input vehicle features.
    
    Parameters:
        model: Trained regression model (RandomForestRegressor)
        input_features: A list or NumPy array containing values for 
                         ['cylinders', 'displacement', 'horsepower', 'weight',
                          'acceleration', 'model_year', 'origin', 'vehicle_age', 'make_label']

    Returns:
        Predicted MPG value
    """
    # Ensure input features are in a DataFrame with the correct column names
    input_columns = ['cylinders', 'displacement', 'horsepower', 'weight',
                     'acceleration', 'model_year', 'origin', 'vehicle_age', 'make_label']
    
    # If input_features is a list or array, convert it to a DataFrame with column names
    if isinstance(input_features, (list, np.ndarray)):
        input_features = pd.DataFrame([input_features], columns=input_columns)
    
    # Predict MPG using the trained model
    mpg_pred = model.predict(input_features)
    return mpg_pred[0]  # Return the predicted MPG value for the input data

# Example usage:
# Assuming you have a trained model 'rf_model' and some input data:
input_data = [6, 230, 130, 3000, 15, 2020, 1, 5, 3]  # Example input values
predicted_mpg = predictMPG(rf_model, input_data)
print(f"Predicted MPG: {predicted_mpg}")

import unittest

def predictMPG(model, input_features):
    """
    Predicts the miles per gallon (MPG) based on input vehicle features.
    
    Parameters:
        model: Trained regression model (RandomForestRegressor)
        input_features: A list or NumPy array containing values for 
                         ['cylinders', 'displacement', 'horsepower', 'weight',
                          'acceleration', 'model_year', 'origin', 'vehicle_age', 'make_label']

    Returns:
        Predicted MPG value
    """
    input_columns = ['cylinders', 'displacement', 'horsepower', 'weight',
                     'acceleration', 'model_year', 'origin', 'vehicle_age', 'make_label']
    
    # If input_features is a list or array, convert it to a DataFrame with column names
    if isinstance(input_features, (list, np.ndarray)):
        input_features = pd.DataFrame([input_features], columns=input_columns)
    
    # Predict MPG using the trained model
    mpg_pred = model.predict(input_features)
    return mpg_pred[0]  # Return the predicted MPG value for the input data

# Unit test class
class TestPredictMPG(unittest.TestCase):
    def setUp(self):
        self.model = RandomForestRegressor(n_estimators=10, random_state=42)
        
        # Create a pandas DataFrame for training data with feature names
        X_dummy = np.random.rand(100, 9)  
        X_dummy = pd.DataFrame(X_dummy, columns=['cylinders', 'displacement', 'horsepower', 'weight',
                                                 'acceleration', 'model_year', 'origin', 'vehicle_age', 'make_label'])
        y_dummy = np.random.rand(100) * 30  # Random MPG values
        self.model.fit(X_dummy, y_dummy)  # Fit model with DataFrame

    def test_prediction_output(self):
        input_features = [6, 250, 100, 3000, 15, 76, 1, 47, 3]  # Example input for the test
        prediction = predictMPG(self.model, input_features)

        self.assertIsInstance(prediction, float)  
        self.assertGreaterEqual(prediction, 0)  
        self.assertLessEqual(prediction, 100) 

# Run tests in Jupyter Notebook
suite = unittest.TestLoader().loadTestsFromTestCase(TestPredictMPG)
unittest.TextTestRunner().run(suite)

import gradio as gr

# Assuming that rf_model is already trained elsewhere
# You should load your trained model if it's already saved
# Dummy training data for illustration (replace with your actual model)
X_train = np.random.rand(100, 9)  # 100 samples, 9 features
X_train = pd.DataFrame(X_train, columns=['cylinders', 'displacement', 'horsepower', 'weight',
                                         'acceleration', 'model_year', 'origin', 'vehicle_age', 'make_label'])
y_train = np.random.rand(100) * 30  # Random MPG values
rf_model = RandomForestRegressor(n_estimators=10, random_state=42)
rf_model.fit(X_train, y_train)

# Function to predict MPG using the trained RandomForest model
def predictMPG(cylinders, displacement, horsepower, weight, acceleration, model_year, origin, vehicle_age, make_label):
    # Convert the input variables to a 2D array in the same order as in the training set
    input_data = np.array([[cylinders, displacement, horsepower, weight, acceleration, model_year, origin, vehicle_age, make_label]])
    
    # Convert input data to DataFrame to match training format
    input_data_df = pd.DataFrame(input_data, columns=['cylinders', 'displacement', 'horsepower', 'weight',
                                                      'acceleration', 'model_year', 'origin', 'vehicle_age', 'make_label'])
    
    # Predict MPG using the trained model
    mpg_pred = rf_model.predict(input_data_df)
    return mpg_pred[0]  # Return the predicted MPG value

# Create the Gradio interface
interface = gr.Interface(
    fn=predictMPG,
    inputs=[
        gr.Number(label="Cylinders"),
        gr.Number(label="Displacement"),
        gr.Number(label="Horsepower"),
        gr.Number(label="Weight"),
        gr.Number(label="Acceleration"),
        gr.Number(label="Model Year"),  # This matches the column name in your dataset
        gr.Number(label="Origin"),
        gr.Number(label="Vehicle Age"),
        gr.Number(label="Make Label")
    ],
    outputs="number",  # Output will be a numeric prediction (MPG)
    title="MPG Prediction",
    description="Enter vehicle attributes to predict miles per gallon (MPG)."
)

# Launch the interface
interface.launch(share=True)
