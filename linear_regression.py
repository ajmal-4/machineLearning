import numpy as np

# Linear Regression Class
class LinearRegression:
    # Initialise the instance of the class with learning_rate, n_iters, weight & bias
    def __init__(self, learning_rate=0.001, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    # Fit function (Training Function) - Receives the input X_train and y_train
    def fit(self, X, y):
        # Find the no.of samples in the dataset and corresponding no.of features
        n_samples, n_features = X.shape
        # Initialising the weights and bias parameters with zero
        self.weights = np.zeros(n_features) #(if zero initialisation is not preferred - use np.random.randn())
        self.bias = 0

        # Gradient descent (Start Training)
        for _ in range(self.n_iters):
            # Predicted output = (Dot product of the input and weights) + bias
            y_predicted = np.dot(X, self.weights) + self.bias

            # Computing the gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # Updating the weights and bias
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    # Prediction (Testing with unseen data)
    def predict(self, X):
        y_approximated = np.dot(X, self.weights) + self.bias
        return y_approximated
    
# Finding mean squared error - Mean of squared distance between Actual Output and Predicted output
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Function to calculate r2 score
def r2_score(y_true, y_pred):
    corr_matrix = np.corrcoef(y_true, y_pred)
    corr = corr_matrix[0, 1]
    return corr ** 2

# Main Function
if __name__ == "__main__":
    # Importing libraries for plotting and dataset
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn import datasets

    # Assign X for data samples and y for corresponding labels
    X, y = datasets.make_regression(
        n_samples=100, n_features=1, noise=20, random_state=4
    )

    # Split the X and y into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1234
    )

    # Crete an instance of the class LinearRegression
    regressor = LinearRegression(learning_rate=0.01, n_iters=1000)
    # Fit the X_train and y_train using the fit function of the class LinearRegression
    regressor.fit(X_train, y_train)
    # Making Predictions using X_test
    predictions = regressor.predict(X_test)

    # Calculating MSE - Mean Squared Error
    mse = mean_squared_error(y_test, predictions)
    print("MSE:", mse)

    # Calculating r2_score
    accu = r2_score(y_test, predictions)
    print("Accuracy:", accu)

    # Plot the figure
    y_pred_line = regressor.predict(X)
    cmap = plt.get_cmap("viridis")
    fig = plt.figure(figsize=(8, 6))
    m1 = plt.scatter(X_train, y_train, color=cmap(0.9), s=10)
    m2 = plt.scatter(X_test, y_test, color=cmap(0.5), s=10)
    plt.plot(X, y_pred_line, color="black", linewidth=2, label="Prediction")
    plt.show()