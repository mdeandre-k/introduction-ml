from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris_dataset = load_iris()

print("keys:\n", iris_dataset.keys())

print(iris_dataset['DESCR'][:193], "\n...")
print("Target names", iris_dataset["target_names"])
print("Feature names", iris_dataset["feature_names"])
print("Type of data", type(iris_dataset["data"]))
print("Shape of data", iris_dataset["data"].shape)
print(iris_dataset['data'][:5])
print("Type of target", type(iris_dataset['target']))
print("Target", iris_dataset['target'])

X_train, X_test, y_train, y_test = \
    train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)

print(len(X_train))
