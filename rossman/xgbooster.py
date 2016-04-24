__author__ = 'Karl'

import pandas as pd
import numpy as np
from sklearn import cross_validation
from time import process_time
from sklearn.cross_validation import train_test_split
import xgboost as xgb


# Thanks to Chenglong Chen for providing this in the forum
def ToWeight(y):
    w = np.zeros(y.shape, dtype=float)
    ind = y != 0
    w[ind] = 1./(y[ind]**2)
    return w


def rmspe(yhat, y):
    w = ToWeight(y)
    rmspe = np.sqrt(np.mean( w * (y - yhat)**2 ))
    return rmspe


def rmspe_xg(yhat, y):
    # y = y.values
    y = y.get_label()
    y = np.exp(y) - 1
    yhat = np.exp(yhat) - 1
    w = ToWeight(y)
    rmspe = np.sqrt(np.mean(w * (y - yhat)**2))
    return "rmspe", rmspe


# Gather some features
def merge(df):
    df = df.merge(store, on='Store', how="left")
    # Some Store have empty open value on test set
    df.loc[df.Open.isnull(), 'Open'] = 1
    # Break down date column
    df['year'] = df.Date.apply(lambda x: x.year)
    df['month'] = df.Date.apply(lambda x: x.month)
    df['woy'] = df.Date.apply(lambda x: x.weekofyear)

    # Calculate time competition open time in months
    df['CompetitionOpen'] = 12 * (df.year - df.CompetitionOpenSinceYear) + (df.month - df.CompetitionOpenSinceMonth)
    df['CompetitionOpen'] = df.CompetitionOpen.apply(lambda x: x if x > 0 else 0)

    # Promo open time in months
    df['PromoOpen'] = 12 * (df.year - df.Promo2SinceYear) + (df.woy - df.Promo2SinceWeek) / float(4)
    df['PromoOpen'] = df.PromoOpen.apply(lambda x: x if x > 0 else 0)
    df['PromoInterval'] = df.PromoInterval.apply(lambda x: x[0] if type(x) == str else 'N')

    df['StateHoliday'] = df.StateHoliday.apply(lambda x: 1 if (x != '0') else 0)

    df = pd.get_dummies(df, columns=[
        'PromoInterval',
        'StoreType',
        'Assortment'
    ])
    df.drop([
        'Date',
        'Open',
        # 'Store',
        'Promo2SinceYear',
        'Promo2SinceWeek',
        'CompetitionOpenSinceMonth',
        'CompetitionOpenSinceYear',
        'year'], axis=1, inplace=True)
    # bug ValueError: all feature_names must be alphanumerics
    df = df.rename(columns=lambda x: x.replace('_', ''))

    return df

# Load data
t = process_time()
data = pd.read_csv('input/train.csv', parse_dates=['Date'], dtype={'StateHoliday': str})
store = pd.read_csv('input/store.csv')
test = pd.read_csv('input/test.csv', parse_dates=['Date'], dtype={'StateHoliday': str})

data = data[:1000]
test = data[:100]

print('training data loaded in ', process_time() - t)

# exclude closed store
data = data[data['Open'] != 0]

# merge
t = process_time()
data = merge(data)
print('training data processed in ', process_time() - t)

# calculate avg of customer for each store
df_customer = data[['Store', 'Customers']].groupby(['Store']).mean().reset_index()
data = data.drop(['Customers'], axis=1)

data = data.merge(df_customer, on='Store', how='left')
data = data.drop(['Store'], axis=1)

# Make sure columns are in right order
X = data.sort_index(axis=1).fillna(0).drop(['Sales'], axis=1).astype(float)
y = data['Sales'].astype(float)

# Prepare Test Set
t = process_time()
test = merge(test)
test = test.merge(df_customer, on='Store')
test = test.drop(['Store'], axis=1).set_index('Id')
X_test_submission = test.sort_index(axis=1).fillna(0).astype(float)
print('test data processed in ', process_time() - t)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=2)

# xgb_model = xgb.XGBRegressor().fit(X_train, y_train)
# y_pred = xgb_model.predict(X_test)
# print("RSMPE: %.3f" % rmspe(y_pred, y_test))

# make submission
t = process_time()
print("start fitting xgb")
xgb_model = xgb.XGBRegressor().fit(X, y)
print("fitting finished after ", process_time() - t)
t = process_time()
y_pred = xgb_model.predict(X_test_submission)
result = pd.DataFrame({'Id': X_test_submission.index.values, 'Sales': y_pred}).set_index('Id')
result = result.sort_index()
result.to_csv('submission.xgb.csv')
result.to_csv('submission.xgb.excel.csv', sep=";", decimal=",")
print('submission created in ', process_time() - t)



# features = X.columns.values
#
# # params = {"objective": "reg:linear",
# #           "eta": 0.3,
# #           "max_depth": 8,
# #           "subsample": 0.7,
# #           "colsample_bytree": 0.7,
# #           "silent": 1
# #           }
# # num_trees = 300
#
# print("Train a XGBoost model")
# val_size = 100000
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=2)
# dtrain = xgb.DMatrix(X_train[features], np.log(y_train + 1))
# dvalid = xgb.DMatrix(X_test, np.log(y_test + 1))
# dtest = xgb.DMatrix(X_test)
# watchlist = [(dvalid, 'eval'), (dtrain, 'train')]
# gbm = xgb.train(params, dtrain, num_trees, evals=watchlist, early_stopping_rounds=50, feval=rmspe_xg, verbose_eval=True)
#
# print("Validating")
# train_probs = gbm.predict(xgb.DMatrix(X_test))
# indices = train_probs < 0
# train_probs[indices] = 0
# error = rmspe(np.exp(train_probs) - 1, X_test['Sales'].values)
# print('error', error)
#
# print("Make predictions on the test set")
# test_probs = gbm.predict(xgb.DMatrix(X_test))
# indices = test_probs < 0
# test_probs[indices] = 0
# submission = pd.DataFrame({"Id": test["Id"], "Sales": np.exp(test_probs) - 1})
# submission.to_csv("xgboost_kscript_submission.csv", index=False)