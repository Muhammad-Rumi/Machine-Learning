import numpy as np

# Step 1: Define the hypothesis function

def hypothesis_function(X, theta):
    # X: input features (matrix)
    # theta: model parameters (vector)
    # Returns the predicted values (vector)
    return theta.T @ X

# Step 2: Define the cost function

def cost_function(y_true, y_pred):
   
    m = len(y)  # Number of training examples
    epsilon = 1e-10 #to prevent from log(0)
    cost = -np.sum(y_true * np.log(y_pred + epsilon) + (1 - y_true) * np.log(1 - y_pred + epsilon)) / 2*m
    return cost

# Step 3: Define the gradient descent algorithm

def gradient_descent(X, y, theta, learning_rate, num_iterations):
    # X: input features (matrix)
    # y: actual output values (vector)
    # theta: model parameters (vector)
    # learning_rate: learning rate for gradient descent
    # num_iterations: number of iterations for gradient descent
    m = len(y)  # Number of training examples
    cost = []  # List to store the cost after each iteration
    sigmoid = lambda a:1/(1+np.exp(-a))
    for _ in range(num_iterations):
        cost_sum = 0
        for i in range(m):
            a = sigmoid(theta.T * X[i])
            error = y[i]-a
            gradient = X.T * error
            theta = theta - learning_rate * gradient
            cost += cost_function(y,a)
        cost.append(cost/m)
    return theta, cost

# Step 4: Prepare the data and run linear regression

# Prepare the input features (X) and output values (y)
X = ...  # Input features as a matrix (each row represents a training example)
y = ...  # Output values as a vector

# Initialize the model parameters (theta)
theta = ...  # Randomly initialize the parameters or use zeros

# Set the learning rate and number of iterations
learning_rate = ...
num_iterations = ...

# Run gradient descent to optimize the model parameters
theta, costs = gradient_descent(X, y, theta, learning_rate, num_iterations)

# Step 5: Evaluate the model and make predictions

# Use the trained model parameters to make predictions on new data
X_test = ...  # New input features for prediction
y_pred = hypothesis_function(X_test, theta)
