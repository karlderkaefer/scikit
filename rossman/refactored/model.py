from time import process_time
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, \
    GradientBoostingRegressor
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import SGDRegressor, Perceptron, \
    LogisticRegression, ElasticNet, RANSACRegressor, \
    LinearRegression, RidgeCV, Ridge
from sklearn.metrics import make_scorer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
import refactored.common as c
import pandas as pd
from scipy.stats import randint as sp_randint
import numpy as np


# save train and test set to local disc
def save_sets():
    train, test, feature_names = c.get_sets(None)
    train.to_pickle("train.pkl")
    test.to_pickle("test.pkl")


def load_sets():
    train = pd.read_pickle("train.pkl")
    test = pd.read_pickle("test.pkl")
    return train, test, c.get_feature_names()


# save_sets()
train, test, feature_names = load_sets()

# need only 5 % to evualate model
X_train, X_test, y_train, y_test = train_test_split(
    train[feature_names], train.Sales, test_size=0.95, random_state=2)


def compare_models():
    estimators = [
        SGDRegressor(loss='squared_loss', penalty='l2'),
        RANSACRegressor(LinearRegression()), ElasticNet(), Ridge(),
        DecisionTreeRegressor(), RandomForestRegressor(),
        GradientBoostingRegressor()
    ]
    estimators_names = [
        "SGDRegressor", "RANSACRegressor", "ElasticNet", "Ridge",
        "DecisionTreeRegressor", "RandomForestRegressor",
        "GradientBoostingRegressor"
    ]
    for estimator, estimator_name in zip(estimators,
                                         estimators_names):
        clf = make_pipeline(StandardScaler(), estimator)
        c.train_score(estimator_name, clf, X_train, y_train)


def optimize_ridge():
    clf = RidgeCV(alphas=[0.1, 1.0, 10.0, 20.0, 100.0],
                  scoring="mean_squared_error")
    clf.fit(X_train, y_train)
    print(clf.alpha_)


def gridsearch():
    custom_scorer = make_scorer(c.rmspe, greater_is_better=False)

    clf = Pipeline([
        ["scl", StandardScaler()],
        ["clf", SGDRegressor()]
    ])
    params = [{
        'clf__loss': ['squared_loss', 'huber', 'epsilon_insensitive',
                      'squared_epsilon_insensitive'],
        'clf__penalty': ['none', 'l2', 'l1', 'elasticnet'],
        'clf__alpha': [000.1, 00.1, 0.1, 1.0, 10, 100]
    }]
    c.print_prediction_err(clf, X_train, y_train,
                           "score before tuning")
    gs = GridSearchCV(
        estimator=clf,
        param_grid=params,
        # wont make sense score needs to be between 0 and 1
        # scoring=custom_scorer,
        scoring="mean_squared_error",
        verbose=True
    )
    gs.fit(X_train, y_train)
    c.print_prediction_err(gs.best_estimator_, X_train, y_train,
                           "score after tuning")
    # {'clf__penalty': 'l1', 'clf__alpha': 0.1, 'clf__loss': 'squared_loss'}
    print("best params: ", gs.best_params_)


def make_prediction_sgd():
    # this will obtain 0.18925 on kaggle
    estimator = SGDRegressor(loss="squared_loss", penalty="l1",
                             alpha=0.1)
    clf = make_pipeline(StandardScaler(), estimator)
    c.submission(clf, train, test, feature_names, "sdg")


def make_prediction_decisiontree():
    # this will obtain 0.17401 on kaggle
    clf = DecisionTreeRegressor()
    c.submission(clf, train, test, feature_names, "decision")


def make_prediction_forest():
    # this will obtain 0.14432 on kaggle
    clf = RandomForestRegressor(n_jobs=-1)
    c.submission(clf, train, test, feature_names, "forest")


def optimize_forest():
    clf = RandomForestRegressor(n_jobs=3)
    params = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [3, None],
        "max_features": sp_randint(1, 11),
        "min_samples_split": sp_randint(1, 11),
        "min_samples_leaf": sp_randint(1, 11),
        "bootstrap": [True, False]
    }
    t = process_time()
    print("starting random grid search for forest..")
    random_cv = RandomizedSearchCV(clf, param_distributions=params,
                                   n_iter=20)
    random_cv.fit(X_train, y_train)
    print("random grid search for forest took", process_time() - t)
    c.report(random_cv.grid_scores_)


def make_prediction_optimizedforest():
    # this will obtain 0.12925 on kaggle
    clf = RandomForestRegressor(
        n_jobs=-1, max_depth=None, max_features=8, bootstrap=False,
        min_samples_leaf=3, n_estimators=100, min_samples_split=8)
    c.submission(clf, train, test, feature_names, "optimizedforest")


def plot_prediction():
    clf = RandomForestRegressor(
        n_jobs=-1, max_depth=None, max_features=8, bootstrap=False,
        min_samples_leaf=3, n_estimators=100, min_samples_split=8)
    c.plot_regression_error(clf, X_train, y_train, "RandomForest")


def plot_partial_dependence():
    clf = GradientBoostingRegressor(n_estimators=100, max_depth=4,
                                    learning_rate=0.1, loss='huber',
                                    random_state=1)
    features = [(4, 22), 2, 3, 4]
    c.plot_gradiant(clf, X_train, y_train, features, feature_names)

# compare_models()
# optimize_ridge()
# gridsearch()
# make_prediction_sgd()
# make_prediction_decisiontree()
# make_prediction_forest()
# optimize_forest()
# make_prediction_optimizedforest()
# plot_prediction()
# plot_partial_dependence()
