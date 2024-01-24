# Install libraries not already in the environment using pip
# !pip install pandas==1.3.4
# !pip install scikit-learn==0.24.1

# Pandas will allow us to create a dataframe of the data so it can be used and manipulated
import pandas as pd
# Regression Tree Algorithm
from sklearn.tree import DecisionTreeRegressor
# Split our data into training and testing sets
from sklearn.model_selection import train_test_split
# File dialog
from tkinter import Tk, filedialog

# Open a file dialog to choose the CSV file
root = Tk()
root.withdraw()  # Hide the main window
file_path = filedialog.askopenfilename(title="Choose the CSV file", filetypes=[("CSV files", "*.csv")])
root.destroy()

# Read the CSV file into a DataFrame
data = pd.read_csv(file_path)

# Display the first few rows of the DataFrame
print(data.head())

# Display the shape of the DataFrame
print(data.shape)

# Check for missing values
print(data.isna().sum())

# Drop rows with missing values
data.dropna(inplace=True)

# Confirm that there are no missing values
print(data.isna().sum())

# Separate features (X) and target variable (Y)
X = data.drop(columns=["MEDV"])
Y = data["MEDV"]

# Display the first few rows of features
print(X.head())

# Display the first few rows of the target variable
print(Y.head())

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, random_state=1)

# Create a regression tree model with squared error criterion
regression_tree = DecisionTreeRegressor(criterion='squared_error')

# Fit the model on the training data
regression_tree.fit(X_train, Y_train)

# Evaluate the model on the testing data
print("R-squared Score:", regression_tree.score(X_test, Y_test))

# Make predictions on the testing data
prediction = regression_tree.predict(X_test)

# Print the mean absolute error in dollars
print("Mean Absolute Error: $", (prediction - Y_test).abs().mean() * 1000)

# Create a regression tree model with friedman_mse criterion
regression_tree_mse = DecisionTreeRegressor(criterion="friedman_mse")

# Fit the model on the training data
regression_tree_mse.fit(X_train, Y_train)

# Evaluate the model on the testing data
print("R-squared Score (MSE):", regression_tree_mse.score(X_test, Y_test))

# Make predictions on the testing data
prediction_mse = regression_tree_mse.predict(X_test)

# Print the mean absolute error in dollars for the MSE model
print("Mean Absolute Error (MSE): $", (prediction_mse - Y_test).abs().mean() * 1000)
