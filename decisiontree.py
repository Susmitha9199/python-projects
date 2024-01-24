# Suppress SSL verification warnings
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import sklearn.tree as tree
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog

# Allow the user to select a file interactively
root = Tk()
root.withdraw()
file_path = filedialog.askopenfilename(title="Select CSV file", filetypes=[("CSV files", "*.csv")])
root.destroy()

# Check if a file was selected
if not file_path:
    print("No file selected. Exiting.")
    exit()

# Load the dataset from the selected file
my_data = pd.read_csv(file_path, delimiter=",")

# Display the first few rows of the dataset
my_data.head()

# Extract features and labels
X = my_data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values

# Label encoding for categorical variables
le_sex = preprocessing.LabelEncoder()
le_sex.fit(['F', 'M'])
X[:, 1] = le_sex.transform(X[:, 1])

le_BP = preprocessing.LabelEncoder()
le_BP.fit(['LOW', 'NORMAL', 'HIGH'])
X[:, 2] = le_BP.transform(X[:, 2])

le_Chol = preprocessing.LabelEncoder()
le_Chol.fit(['NORMAL', 'HIGH'])
X[:, 3] = le_Chol.transform(X[:, 3])

# Extract labels
y = my_data["Drug"]

# Split the dataset into training and testing sets
X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=3)

# Display the shapes of training and testing sets
print('Shape of X training set:', X_trainset.shape, '& Size of Y training set:', y_trainset.shape)
print('Shape of X testing set:', X_testset.shape, '& Size of Y testing set:', y_testset.shape)

# Create a Decision Tree classifier
drugTree = DecisionTreeClassifier(criterion="entropy", max_depth=4)

# Train the model
drugTree.fit(X_trainset, y_trainset)

# Make predictions on the testing set
predTree = drugTree.predict(X_testset)

# Display some predictions and actual values
print('Predictions:', predTree[:5])
print('Actual values:', y_testset[:5])

# Evaluate the model's accuracy
accuracy = metrics.accuracy_score(y_testset, predTree)
print("Decision Tree's Accuracy:", accuracy)

# Plot the Decision Tree
tree.plot_tree(drugTree)
plt.show()
