# project1
# Import necessary libraries
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load the data into a pandas dataframe
import pandas as pd
data = pd.read_csv("cement_data.csv")

# Split the data into features and target
X = data[['type', 'raw_material_1', 'raw_material_2', 'raw_material_3']]
y = data['compressive_strength']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an instance of the Linear Regression model
reg = LinearRegression()

# Train the model on the training data
reg.fit(X_train, y_train)

# Make predictions on the test data
y_pred = reg.predict(X_test)

# Calculate the model's accuracy
from sklearn.metrics import r2_score
accuracy = r2_score(y_test, y_pred)
print("Accuracy:", accuracy)
