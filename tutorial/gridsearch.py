from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.grid_search import GridSearchCV
import numpy as np
from sklearn.metrics import mean_squared_error

import common as c

X, y = c.get_set()

# # we don't need to scale data
# clf = RandomForestRegressor(n_jobs=-1)
# scores = cross_val_score(clf, X, y, scoring="mean_squared_error")
# print("%.3f +/- %.3f = MSE before tuning"
#          % (np.mean(scores), np.std(scores)))
#
# params = [{
#     'max_depth': [30, 70, 100],
#     'n_estimators': [10, 100, 200]
# }]
# gs = GridSearchCV(
#     estimator=clf,
#     param_grid=params,
#     cv=5,
#     scoring="mean_squared_error"
# )
# gs.fit(X, y)
# print("best params: ", gs.best_params_)
# scores = cross_val_score(gs.best_estimator_, X, y,
#                          scoring="mean_squared_error", cv=5)
# print("%.3f +/- %.3f = MSE after tuning"
#          % (np.mean(scores), np.std(scores)))
#
# # we found optimzed parameter
# clf = RandomForestRegressor(max_depth=70, n_estimators=100)
# c.plot_prediction_error("RandomForestRegressorOptimized", clf, X, y)

# instead of crossvalidation we use out-of-bag prediction
clf = RandomForestRegressor(oob_score=True,
                            n_estimators=100,
                            max_depth=100)
clf.fit(X, y)
prediction = clf.oob_prediction_
c.plot("RandomForestOOB", prediction, y)
