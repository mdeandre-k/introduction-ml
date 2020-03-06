import mglearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

X, y = mglearn.datasets.load_extended_boston()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
lr = LinearRegression().fit(X_train, y_train)

print("Training score", lr.score(X_train, y_train))
print("Test score", lr.score(X_test, y_test))


ridge = Ridge(alpha=0.1).fit(X_train, y_train)
print("Training score", ridge.score(X_train, y_train))
print("Test score", ridge.score(X_test, y_test))

lasso = Lasso(alpha=0.01, max_iter=100000).fit(X_train, y_train)
print("Train lasso", lasso.score(X_train, y_train))
print("Test lasso", lasso.score(X_test, y_test))