from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV

import refactored.common as c
import pandas as pd
import xgboost as xgb
import numpy as np
import operator
import matplotlib
matplotlib.use("Agg")  # Needed to save figures
import matplotlib.pyplot as plt

# save train and test set to local disc
def save_sets():
    train, test, feature_names = c.get_sets(None)
    train.to_pickle("train.boost.pkl")
    test.to_pickle("test.boost.pkl")

def load_sets():
    train = pd.read_pickle("train.boost.pkl")
    test = pd.read_pickle("test.boost.pkl")
    return train, test, c.get_feature_names()


# save_sets()
train, test, feature_names = load_sets()

# need only 5 % to evualate model
X_train, X_test, y_train, y_test = train_test_split(
    train[feature_names], train.Sales, test_size=0.999, random_state=2)


def gridsearch():
    # for parallel execution add n_jobs=-1 in gridsearchcv
    # from https://github.com/dmlc/xgboost/blob/master/demo/guide-python/sklearn_parallel.py
    xgb_model = xgb.XGBRegressor()
    print(X_train.shape)
    params = {'max_depth': [2, 4, 6],
              'n_estimators': [50, 100, 200]
              }
    clf = GridSearchCV(xgb_model, params,
                       verbose=2,
                       scoring="mean_squared_error")
    clf.fit(X_train, y_train)
    print(clf.best_score_)
    print(clf.best_params_)


def make_prediction():
    # duration: 15 min, score: 0.14644
    clf = xgb.XGBRegressor(n_estimators=100, max_depth=2)
    c.submission(clf, train, test, feature_names, "xgb")


# save the model on local disc to reuse
def ceate_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    for i, feat in enumerate(features):
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
    outfile.close()


# this will obtain 0.1187 on kaggle
def xgboost_validset_submission():
    params = {"objective": "reg:linear",
          "eta": 0.3,
          "max_depth": 10,
          "subsample": 0.7,
          "colsample_bytree": 0.7,
          "silent": 1,
          "seed": 1301
    }
    num_boost_round = 300
    # need to split for a small validation set
    X_train_xgb, X_valid_xgb = train_test_split(train, test_size=0.012)
    y_train_xgb = np.log1p(X_train_xgb.Sales)
    y_valid_xgb = np.log1p(X_valid_xgb.Sales)
    dtrain = xgb.DMatrix(X_train_xgb[feature_names], y_train_xgb)
    dvalid = xgb.DMatrix(X_valid_xgb[feature_names], y_valid_xgb)

    watchlist = [(dvalid, 'eval'), (dtrain, 'train')]
    gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist,
                    early_stopping_rounds=100,
                    feval=c.rmspe_xg, verbose_eval=True)

    print("Validating")
    y_pred = gbm.predict(xgb.DMatrix(X_valid_xgb[feature_names]))
    error = c.rmspe(X_valid_xgb.Sales.values, np.expm1(y_pred))
    print('RMSPE: {:.6f}'.format(error))

    print("Make predictions on the test set")
    dtest = xgb.DMatrix(test[feature_names])
    test_probs = gbm.predict(dtest)
    # Make Submission
    result = pd.DataFrame({"Id": test["Id"],
                           'Sales': np.expm1(test_probs)})
    result.to_csv("xgboost_10_submission.csv", index=False)

    # XGB feature importances
    # Based on https://www.kaggle.com/mmueller/
    # liberty-mutual-group-property-inspection-prediction/
    # xgb-feature-importance-python/code

    ceate_feature_map(feature_names)
    importance = gbm.get_fscore(fmap='xgb.fmap')
    importance = sorted(importance.items(), key=operator.itemgetter(1))

    df = pd.DataFrame(importance, columns=['feature', 'fscore'])
    df['fscore'] = df['fscore'] / df['fscore'].sum()

    featp = df.plot(kind='barh', x='feature', y='fscore',
                    legend=False, figsize=(6, 10))
    plt.title('XGBoost Feature Importance')
    plt.xlabel('relative importance')
    fig_featp = featp.get_figure()
    fig_featp.savefig('feature_importance_xgb.png',
                      bbox_inches='tight', pad_inches=1)


# gridsearch
# make_prediction()

