import numpy as np

# Creating a class - Perceptron
class Perceptron:
    # Initiating the instance of the class with learning_rate, no.of iterations and activation function
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.activation_func = self._unit_step_func
        self.weights = None
        self.bias = None

    # Fitting the model with training data
    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Initialising weight and bias parameters - Initialising with zeros
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Converting the target data(y) into 1 or 0 (2 classes)
        y_ = np.array([1 if i > 0 else 0 for i in y])

        # No.of iterations as per given
        for _ in range(self.n_iters):

            # Iterate through each sample
            for idx, x_i in enumerate(X):

                # Linear function (wTx+b)
                linear_output = np.dot(x_i, self.weights) + self.bias
                # Putting actiavtion function
                y_predicted = self.activation_func(linear_output)

                # Perceptron update rule
                update = self.lr * (y_[idx] - y_predicted)

                self.weights += update * x_i
                self.bias += update

    # Function to predict the unseen data
    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self.activation_func(linear_output)
        return y_predicted

    # Unit step function - 1 for non negative functions and 0 for negative
    def _unit_step_func(self, x):
        return np.where(x >= 0, 1, 0)


# Testing - Main Function
if __name__ == "__main__":
    # Imports for plotting and dataset
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn import datasets

    # Convert the dataset into training and testing datasets
    X, y = datasets.make_blobs(
        n_samples=150, n_features=2, centers=2, cluster_std=1.05, random_state=2
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123
    )

    # Create an instance of the class - Perceptron
    p = Perceptron(learning_rate=0.01, n_iters=1000)
    # Fit the data into the model
    p.fit(X_train, y_train)
    # Predict the test data
    predictions = p.predict(X_test)

    # Calculate the accuracy of the model
    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy
    print("Perceptron classification accuracy", accuracy(y_test, predictions))

    # Plot the figure
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.scatter(X_train[:, 0], X_train[:, 1], marker="o", c=y_train)

    x0_1 = np.amin(X_train[:, 0])
    x0_2 = np.amax(X_train[:, 0])

    x1_1 = (-p.weights[0] * x0_1 - p.bias) / p.weights[1]
    x1_2 = (-p.weights[0] * x0_2 - p.bias) / p.weights[1]

    ax.plot([x0_1, x0_2], [x1_1, x1_2], "k")

    ymin = np.amin(X_train[:, 1])
    ymax = np.amax(X_train[:, 1])
    ax.set_ylim([ymin - 3, ymax + 3])

    plt.show()