from sklearn.datasets import load_iris
iris = load_iris()
# meta data's from the set
print(iris.feature_names)
print(iris.target_names)
print(iris.data[0])
print(iris.target[0])

# iterating over features and displaying
for i in range(len(iris.target)):
    print("Example %d: label %s, features %s" % (i, iris.target[i], iris.data[i]))
