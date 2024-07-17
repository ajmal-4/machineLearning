import numpy as np
from collections import Counter

# To calculate euclidean distance between two points
def euclidean_distance(x1,x2):
    return np.sqrt(np.sum((x1-x2)**2))

# Create a KNN class
class KNN:

    # Initialise the instance with K value
    def __init__(self,k=3):
        self.k = k

    # Fit Function (Training Function)
    def fit(self, X, y):
        # Assign training data to the instance
        self.X_train = X
        self.y_train = y

    # Predict Function
    def predict(self, X):
        predicted_labels = [self._predict(x) for x in X]
        return np.array(predicted_labels)

    def _predict(self, x):
        # Computing distances of a point with repect to each point
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        # Get k nearest samples, labels
        k_indices = np.argsort(distances)[:self.k]
        # Extract the labels
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        # Majority vote
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

# Main Function
if __name__ == "__main__":
    # Importing libraries for plotting and dataset
    from matplotlib.colors import ListedColormap
    from sklearn import datasets
    from sklearn.model_selection import train_test_split

    cmap = ListedColormap(["#FF0000", "#00FF00", "#0000FF"])

    # Load the iris dataset
    iris = datasets.load_iris()
    X, y = iris.data, iris.target

    # Split the data into training and test data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1234
    )

    # Assign the value of K (No.of nearest neighbours)
    k = 3
    # Create an instance of class KNN
    clf = KNN(k=k)
    # Fit the training data to the KNN model
    clf.fit(X_train, y_train)
    # Make predictions using test dataset
    predictions = clf.predict(X_test)

    # Function to find the accuracy of the model
    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy
    # Print the accuracy of the model
    print("KNN classification accuracy", accuracy(y_test, predictions))
