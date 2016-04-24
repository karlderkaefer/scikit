import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, BaggingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import SGDRegressor, LogisticRegression, Ridge, TheilSenRegressor, Lasso, ElasticNet
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeRegressor

df = pd.read_csv('input/train.csv', parse_dates=['datetime'])
df['year'] = df.datetime.apply(lambda x: x.year)
df['month'] = df.datetime.apply(lambda x: x.month)
df['dow'] = df.datetime.apply(lambda x: x.dayofweek)
df['woy'] = df.datetime.apply(lambda x: x.weekofyear)
df['hour'] = df.datetime.apply(lambda x: x.hour)
df = pd.get_dummies(df, columns=['season', 'dow'])
# we wont use causal and registered in tutorial
df.drop(['registered', 'casual'], axis=1, inplace=True)
df.drop(['datetime'], axis=1, inplace=True)

from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import cross_val_predict
from sklearn.metrics import mean_squared_error
import seaborn as sns
import numpy as np
import pylab as plt

# X are our features without 'count'
X = df.drop(['count'], axis=1)
# y is the target 'count'
y = df['count']
# test and train split will be done by cross validation
# X_train, X_test, y_train, y_test = \
#    train_test_split(X, y, test_size=0.70, random_state=2)

def kfold(name, clf, X_train, y_train):
    print("------------- ", name, " --------------")
    clf.fit(X, y)
    print("%.3f = Coefficient of determination on training set"
          % clf.score(X_train, y_train))
    cv = KFold(X_train.shape[0], 5, shuffle=True)
    scores = cross_val_score(clf, X_train, y_train, cv=cv)
    print("%.3f +/- %.3f = Cross validation k-Fold k=5 accuracy"
          % (np.mean(scores), np.std(scores)))

clf = make_pipeline(StandardScaler(), SGDRegressor())
kfold("SDGRegressor", clf, X, y)

def plot_prediction_error(name, clf, X, y):
    plt.figure()
    cv = KFold(X.shape[0], 5, shuffle=True)
    predicted = cross_val_predict(clf, X, y, cv=cv)
    print("%.3f = mean squared error" % mean_squared_error(y, predicted))
    sns.regplot(x=y[:1000], y=predicted[:1000])
    sns.axlabel("actual", "predicted")
    plt.savefig("plot_validation_" + name + ".png")


plot_prediction_error("SDGRegressor", clf, X, y)

estimators = [
    SGDRegressor(), Lasso(), Ridge(), ElasticNet(),
    DecisionTreeRegressor(), RandomForestRegressor(),
    GradientBoostingRegressor()
]
estimators_names = [
    "SGDRegressor", "Lasso", "Ridge", "ElasticNet",
    "DecisionTreeRegressor", "RandomForestRegressor",
    "GradientBoostingRegressor"
]
for estimator, estimator_name in zip(estimators, estimators_names):
    clf = make_pipeline(StandardScaler(), estimator)
    kfold(estimator_name, clf, X, y)
    plot_prediction_error(estimator_name, clf, X, y)






