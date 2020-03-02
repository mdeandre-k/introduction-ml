from sklearn.datasets import load_breast_cancer
import numpy as np

cancer = load_breast_cancer()
print("Cancer keys:", cancer.keys())
print("Shape of cancer data", cancer.data.shape)
print("Sample count per class:",
      {n: v for n, v in zip(cancer.target_names, np.bincount(cancer.target))})
print("Feature names:", cancer.feature_names)

