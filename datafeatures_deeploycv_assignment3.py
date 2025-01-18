import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# User input for the number of features and data points
num_features = int(input("Enter number of features: "))
num_points = int(input("Enter number of data points: "))

# Input the dataset of features and data points
X = []
y = []

print("Enter the data points (each feature separated by space, followed by the target value):")
for i in range(num_points):
    row = list(map(float, input(f"Enter data point {i+1}: ").split()))
    X.append(row[:-1])  # All except last value as feature values
    y.append(row[-1])   # The last value as the target value

X = np.array(X)
y = np.array(y)

# Check if the number of features matches the dataset dimensions
if X.shape[1] != num_features:
    print(f"Warning: The number of features you entered ({num_features}) does not match the dataset's number of features.")

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Linear Regression model on the training data
model = LinearRegression()
model.fit(X_train, y_train)

# Print the coefficients and the intercept (equation of the plane)
print(f"Coefficients (slope of each feature): {model.coef_}")
print(f"Intercept (constant term): {model.intercept_}")

# Visualizing the best fit plane (only for 2 features)
if num_features == 2:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot of the training data points
    ax.scatter(X_train[:, 0], X_train[:, 1], y_train, color='r', marker='o')

    # Create a grid for plotting the plane
    x_range = np.linspace(min(X_train[:, 0]), max(X_train[:, 0]), 100)
    y_range = np.linspace(min(X_train[:, 1]), max(X_train[:, 1]), 100)
    X1, X2 = np.meshgrid(x_range, y_range)
    Z = model.intercept_ + model.coef_[0] * X1 + model.coef_[1] * X2

    # Plot the best fit plane
    ax.plot_surface(X1, X2, Z, color='b', alpha=0.5)

    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_zlabel('Target')

    plt.show()

# Test the model on the testing data
y_pred = model.predict(X_test)

# Output the model's predictions for the testing data
print("\nTesting the model:")
print(f"Actual values (testing data): {y_test}")
print(f"Predicted values (testing data): {y_pred}")

# Calculate and print the performance metrics (e.g., Mean Squared Error and R-squared)
from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R²): {r2}")

# Now, make predictions for new data points (features only)
print("\nEnter new data points (features only) to predict the target value:")
while True:
    new_point = input(f"Enter new data point (features only) or type 'exit' to quit: ")
    if new_point.lower() == 'exit':
        break
    point = list(map(float, new_point.split()))

    # Predict the target value for the new data point
    prediction = model.predict([point])  # Use a 2D array for prediction
    print(f"Predicted target value: {prediction[0]}")

