from sklearn.cross_validation import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, \
    AdaBoostRegressor
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
import common as c
import xgboost as xgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

X, y, features = c.get_full_set()
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.6, random_state=1)

def simple_model():
    xgb_model = xgb.XGBRegressor()
    xgb_model.fit(X_train[features], y_train.values)
    y_pred = xgb_model.predict(X_test[features])
    print("R2 Score: ", r2_score(y_test, y_pred))

def cross_validation():
    xgb_model = xgb.XGBRegressor()
    dtrain = xgb.DMatrix(X_train[features], y_train)
    # http://rpackages.ianhowson.com/cran/xgboost/man/xgb.train.html
    param = {'max_depth':4, 'eta':1, 'silent':1}
    num_round = 20
    score = xgb.cv(params=param, num_boost_round=num_round,
                   nfold=5, metrics={'rmse', 'error'},
                   dtrain=dtrain)
    print(score)

def early_stopping():
    X_train_xgb, X_valid_xgb, y_train_xgb, y_valid_xgb = \
        train_test_split(X_train, y_train, test_size=0.012)
    xgb_model = xgb.XGBRegressor()
    dtrain = xgb.DMatrix(X_train_xgb[features], y_train_xgb.values)
    dvalid = xgb.DMatrix(X_valid_xgb[features], y_valid_xgb.values)
    watchlist = [(dvalid, 'eval'), (dtrain, 'train')]
    num_boost_round = 400
    params = {'max_depth':4, 'silent':1}
    gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist,
                    early_stopping_rounds=100)

    y_pred = gbm.predict(xgb.DMatrix(X_test[features]),
                         ntree_limit=gbm.best_ntree_limit)
    print("R2 Score: ", r2_score(y_test, y_pred))
    xgb.plot_importance(gbm)
    xgb.plot_tree(gbm, num_trees=2)
    plt.show()
    # save model for reuse after gbm
    gbm.save_model("xgb_train.model")


def plot_feature_importance():
    xgb_model = xgb.Booster()
    xgb_model.load_model("xgb_train.model").feature_names(features)
    xgb.plot_importance(xgb_model)
    plt.show()


def compare_gradient_boosting():
    xgb_model = xgb.Booster()
    xgb_model.load_model("xgb_train.model")
    y_pred_xgb = xgb_model.predict(xgb.DMatrix(X_test[features]))

    # fit scikit gradient boosting regressor
    clf = GradientBoostingRegressor(max_depth=4, n_estimators=300)
    clf.fit(X_train[features], y_train)
    y_pred_gb = clf.predict(X_test[features])

    clf2 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4),
                             n_estimators=300)
    clf2.fit(X_train[features], y_train)
    y_pred_ada = clf2.predict(X_test[features])

    labels = ['real', 'XGB', 'GB', 'Ada']
    predicted = [y_test, y_pred_xgb, y_pred_gb, y_pred_ada]
    plot_timeline(predicted, labels, 100)


def plot_timeline(predicted, labels, max_elements):
    df = pd.DataFrame()
    for y_pred, label in zip(predicted, labels):
        df[label] = y_pred
        print("R2 Score: %0.5f  %s" % (r2_score(y_test, y_pred),
                                       label))

    x_value = X_test['datetime'][:max_elements]
    y_value = df[:max_elements]

    plt.plot_date(x_value, y_value)
    plt.legend(labels)
    for i, txt in enumerate(y_value.index.values):
        for label in labels:
            plt.annotate(txt, (x_value.values[i],
                           y_value[label].values[i]))
    plt.show()

# simple_model()
# early_stopping()
compare_gradient_boosting()



