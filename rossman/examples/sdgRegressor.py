import numpy as np
import matplotlib as plt
from sklearn.datasets import load_boston
from sklearn.cross_validation import *
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor


def train_and_evaluate(clf, X, y):
    clf.fit(X, y)
    print("Coefficient of determination on training set: ", clf.score(X_train, y_train))
    cv = KFold(X.shape[0], 5, shuffle=True)
    scores = cross_val_score(clf, X, y, cv=cv)
    print("Average coefficient of determination using 5-fold cross validation: ", np.mean(scores))

boston = load_boston()
print(boston.target)

X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target, test_size=0.25)
scalerX = StandardScaler().fit(X_train)
scalerY = StandardScaler().fit(y_train)

X_train = scalerX.transform(X_train)
y_train = scalerY.transform(y_train)
X_test = scalerX.transform(X_test)
y_test = scalerY.transform(y_test)

print("--------- SVR --------")
clf_svr = SVR(kernel='rbf')
train_and_evaluate(clf_svr, X_train, y_train)

print("--------- SGDRegressor --------")
clf_sgd = SGDRegressor(loss='squared_loss', penalty='l2')
train_and_evaluate(clf_sgd, X_train, y_train)

print("--------- Decision Tree --------")
clf_tree = DecisionTreeRegressor(max_depth=3)
train_and_evaluate(clf_tree, boston.data, boston.target)



