import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree
iris = load_iris()

# meta data's from the set
# print(iris.feature_names)
# print(iris.target_names)
# print(iris.data[0])
# print(iris.target[0])

# iterating over features and displaying
# for i in range(len(iris.target)):
#     print("Example %d: label %s, features %s" % (i, iris.target[i], iris.data[i]))

# setting apart testing ids from the actual set
test_ids = [0, 55, 105]

# training data
train_target = np.delete(iris.target, test_ids)
train_data = np.delete(iris.data, test_ids, axis=0)

# testing data
test_target = iris.target[test_ids]
test_data = iris.data[test_ids]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(train_data, train_target)

# system prediction from the test data
print(clf.predict(test_data))
