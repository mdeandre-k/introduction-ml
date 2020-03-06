from sklearn.datasets import load_breast_cancer
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt


cancer = load_breast_cancer()
print("Cancer keys:", cancer.keys())
print("Shape of cancer data", cancer.data.shape)
print("Sample count per class:",
      {n: v for n, v in zip(cancer.target_names, np.bincount(cancer.target))})
print("Feature names:", cancer.feature_names)

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=66)

# training_accuracy = []
# test_accuracy = []
#
# neighbors_settings = range(1, 11)
# for n_neighbors in neighbors_settings:
#     clf = KNeighborsClassifier(n_neighbors=n_neighbors)
#     clf.fit(X_train, y_train)
#     print(f"For {n_neighbors} neighbors")
#     print(f"Train accuracy: {clf.score(X_train, y_train)}")
#     print(f"Test accuracy: {clf.score(X_test, y_test)}")
#     training_accuracy.append(clf.score(X_train, y_train))
#     test_accuracy.append(clf.score(X_test, y_test))
#
# plt.plot(neighbors_settings, training_accuracy, label="Training accuracy")
# plt.plot(neighbors_settings, test_accuracy, label="Testing accuracy")
# plt.ylabel("Accuracy")
# plt.xlabel("Neighbor number")
# plt.legend()
# plt.show()

logreg = LogisticRegression().fit(X_train, y_train)
print("Train score", logreg.score(X_train, y_train))
print("Test score", logreg.score(X_test, y_test ))