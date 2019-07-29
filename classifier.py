from scipy.spatial import distance

# Euclidean function calculate for distance
def euc(a, b):
    return distance.euclidean(a, b)

# Custom Classifier
class ScrappyKnn:
    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def predict(self, x_test):
        predictions = []
        for row in x_test:
            label = self.closest(row)
            predictions.append(label)
        return predictions

    # Checking the closest point in comparison with training data and test data
    def closest(self, row):
        best_dist = euc(row,self.x_train[0])
        best_index = 0
        for i in range(1, len(self.x_train)):
            dist = euc(row, self.x_train[i])
            if dist < best_dist:
                best_dist = dist
                best_index = i
        return self.y_train[best_index]


from sklearn import datasets
from sklearn.model_selection import train_test_split
iris = datasets.load_iris()

x = iris.data
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5)

# Creating Custom Classifier

clf = ScrappyKnn()
clf.fit(x_train, y_train)
predictions = clf.predict(x_test)

# To find the accuracy score

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, predictions))
