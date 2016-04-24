from datetime import datetime
from pandas.lib import to_datetime
from pandas.tslib import Timestamp
from scipy.spatial.distance import cdist, pdist
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.ensemble import AdaBoostRegressor
import numpy as np
from sklearn.metrics import make_scorer, mean_squared_error
from math import sqrt
from sklearn.tree import DecisionTreeRegressor
import common as c
import matplotlib.pyplot as plt
import pandas as pd

X, y = c.get_set()
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.3, random_state=1)

clf1 = DecisionTreeRegressor(max_depth=4)
clf2 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4))
clf3 = DecisionTreeRegressor(max_depth=300)
clf4 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=300))
estimators = [clf1, clf2, clf3, clf4]

labels = ["DecisionTreeDepth4", "AdaBoostDepth4",
          "DecisionTreeDepth300", "AdaBoostDepth300"]

def compare_models():
    for clf, label in zip(estimators, labels):
            scores = cross_val_score(clf, X_train, y_train,
                                     scoring="r2")
            print("R2 Score: %0.2f (+/- %0.3f) [%s]" %
                  (scores.mean(), scores.std(), label))

def plot_timeline():
    X, y, features = c.get_full_set()
    # get set additionally with datetime
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.6, random_state=1)

    df = pd.DataFrame({"expected": y_test.values})

    for clf, label in zip(estimators, labels):
        clf.fit(X_train[features], y_train)
        y_pred = clf.predict(X_test[features])
        df[label] = y_pred

    print(df.head())
    plt.plot_date(X_test['datetime'], df)
    labels_all = np.insert(labels, 0, 'real')
    plt.legend(labels_all)
    plt.show()

# compare_models()
plot_timeline()



# clf1 = DecisionTreeRegressor(max_depth=4)
# clf2 = DecisionTreeRegressor(max_depth=300)
# clf3 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=300))
# clf1.fit(X_train[features], y_train)
# clf2.fit(X_train[features], y_train)
# y_pred = clf1.predict(X_test[features])
# y_pred2 = clf2.predict(X_test[features])
#
#
# df = pd.DataFrame({
#     "prediction1": y_pred,
#     "prediction2:": y_pred2,
#     "expected": y_test.values
# })
#
# print(df.head())
# # df["diff"] = df["prediction"] - df["expected"]
# plt.plot_date(X_test['datetime'], df)
# plt.legend(['DecisionTree4', 'AdaBoost', 'Real'])
# plt.show()



# print(y_pred_df[0])
# print(df.head())
# print(y_test.head())

#
#
# print(df_diff)

#
# serie = pd.Series(data=y_pred, index=X_test['datetime'])
# plt.plot_date(X_test['datetime'], error)
# plt.show()

