from sklearn import datasets
from sklearn.model_selection import train_test_split
iris = datasets.load_iris()

x = iris.data
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5)

# Decision tree based prediction

# from sklearn import tree
# clf = tree.DecisionTreeClassifier()
# clf = clf.fit(x_train, y_train)
# predictions = clf.predict(x_test)

# Nearest Neighbor based prediction

from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier()
clf = clf.fit(x_train, y_train)
predictions = clf.predict(x_test)

# To find the accuracy score

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, predictions))
