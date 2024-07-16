import numpy as np

# Logistic Regression class
class LogisticRegression:
    
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

        # Initialising the weights and bias parameters with zeros
        self.weights = np.zeros(n_features) #(if zero initialisation is not preferred - use np.random.randn())
        self.bias = 0

        # Gradient descent (Start Training)
        for _ in range(self.n_iters):
            # Approximate y with linear combination of weights and x, plus bias
            linear_model = np.dot(X, self.weights) + self.bias
            # Apply sigmoid function
            y_predicted = self._sigmoid(linear_model)

            # Computing the gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            # Updating the weights and bias
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    # Signmoid Activation function
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # Prediction (Testing with unseen data)
    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_predicted_cls)


# Main Function
if __name__ == "__main__":
    # Importing libraries for dataset
    from sklearn.model_selection import train_test_split
    from sklearn import datasets

    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

    # Assign X for data samples and y for corresponding targets
    bc = datasets.load_breast_cancer()
    X, y = bc.data, bc.target

    # Split the X and y into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1234
    )

    # Crete an instance of the class LogisticRegression
    regressor = LogisticRegression(learning_rate=0.0001, n_iters=1000)
    # Fit the X_train and y_train using the fit function of the class LinearRegression
    regressor.fit(X_train, y_train)
    # Making Predictions using X_test
    predictions = regressor.predict(X_test)

    # Printing the accuracy
    print("LR classification accuracy:", accuracy(y_test, predictions))