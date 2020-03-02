from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import matplotlib.pyplot as plt
import mglearn
import numpy as np

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

iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), marker='o', hist_kwds={'bins': 20}, s=60, alpha=0.8, cmap=mglearn.cm3)
# plt.show()

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
X_new = np.array([[5, 2.9, 1, 0.2]])
print("X_new.shape:", X_new.shape)

prediction = knn.predict(X_new)
print("Prediction:", prediction)
print("Prediction target name:", iris_dataset['target_names'][prediction])

y_predict = knn.predict(X_test)
print("Prediction:", y_predict)
print("Test score: {:.2f}".format(knn.score(X_test, y_test)))