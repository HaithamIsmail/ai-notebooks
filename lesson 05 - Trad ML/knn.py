import numpy as np
from collections import Counter

def l2_distance(x1, x2):
    return np.sqrt(np.sum((x1-x2)**2))

class KNN:
    def __init__(self, k=3):
        self.k = k
        
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    
    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return predictions
        
    def _predict(self, x):
        # compute the distance
        distances = [l2_distance(x, x_train) for x_train in self.X_train]
        
        # get the k closest samples
        nearest_indices = np.argsort(distances)[:self.k]
        nearest_labels = [self.y_train[i] for i in nearest_indices]
        
        # majortiy vote
        most_common = Counter(nearest_labels).most_common()
        return most_common[0][0]

if __name__ == "__main__":
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    
    cmap = ListedColormap(["#FF0000", '#00FF00', "#0000FF"])
    
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    plt.figure()
    plt.scatter(X[:, 2], X[:, 3], c=y, cmap=cmap, edgecolors='k', s=20)
    plt.show()
    
    clf = KNN(k=5)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    accuracy = np.sum(preds == y_test) / len(preds)
    print('accuracy', accuracy)