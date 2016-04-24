from sklearn.utils.sparsefuncs import inplace_column_scale

__author__ = 'Karl'

import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
import xgboost as xgb
import operator
import matplotlib

matplotlib.use("Agg")  # Needed to save figures
import matplotlib.pyplot as plt
from time import process_time


def ceate_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    for i, feat in enumerate(features):
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
    outfile.close()


def rmspe(y, yhat):
    return np.sqrt(np.mean((yhat / y - 1) ** 2))


def rmspe_xg(yhat, y):
    y = np.expm1(y.get_label())
    yhat = np.expm1(yhat)
    return "rmspe", rmspe(y, yhat)


def toBinary(featureCol, df):
    values = set(df[featureCol].unique())
    newCol = [featureCol + val for val in values]
    for val in values:
        df[featureCol + val] = df[featureCol].map(lambda x: 1 if x == val else 0)
    return newCol


# Gather some features
def build_features(data):
    t = process_time()
    # remove NaNs
    data.fillna(0, inplace=True)
    data.loc[data.Open.isnull(), 'Open'] = 0

    # Break down date column
    data['year'] = data.Date.apply(lambda x: x.year)
    data['month'] = data.Date.apply(lambda x: x.month)
    data['woy'] = data.Date.apply(lambda x: x.weekofyear)

    # Calculate time competition open time in months
    data['CompetitionOpen'] = 12 * (data.year - data.CompetitionOpenSinceYear) + (
    data.month - data.CompetitionOpenSinceMonth)
    data['CompetitionOpen'] = data.CompetitionOpen.apply(lambda x: x if x > 0 else 0)

    # Promo open time in months
    data['PromoOpen'] = 12 * (data.year - data.Promo2SinceYear) + (data.woy - data.Promo2SinceWeek) / float(4)
    data['PromoOpen'] = data.PromoOpen.apply(lambda x: x if x > 0 else 0)
    data['PromoInterval'] = data.PromoInterval.apply(lambda x: x[0] if type(x) == str else 'N')

    data['StateHoliday'] = data.StateHoliday.apply(lambda x: 1 if (x != '0') else 0)

    data = pd.get_dummies(data, columns=[
        'PromoInterval',
        'StoreType',
        'Assortment'
    ])

    data.drop([
        'Date',
        'Open',
        # 'Store',
        'Promo2SinceYear',
        'Promo2SinceWeek',
        'CompetitionOpenSinceMonth',
        'CompetitionOpenSinceYear',
        'year'], axis=1, inplace=True)
    # bug ValueError: all feature_names must be alphanumerics
    data = data.rename(columns=lambda x: x.replace('_', ''))
    print('merged in ', process_time() - t)
    return data


## Start of main script

print("Load the training, test and store data using pandas")
train = pd.read_csv("../input/train.csv", parse_dates=['Date'], dtype={'StateHoliday': str})
test = pd.read_csv("../input/test.csv", parse_dates=['Date'], dtype={'StateHoliday': str})
store = pd.read_csv("../input/store.csv")

# train = train[train['Store'] == 1]
# test = test[test['Store'] == 1]

print("Assume store open, if not provided")
train.fillna(1, inplace=True)
test.fillna(1, inplace=True)

print("Consider only open stores for training. Closed stores wont count into the score.")
train = train[train["Open"] != 0]
print("Use only Sales bigger then zero. Simplifies calculation of rmspe")
train = train[train["Sales"] > 0]

print("Join with store")
train = pd.merge(train, store, on='Store', how='left')
test = pd.merge(test, store, on='Store', how='left')

print("augment features")
train = build_features(train)

grouped_columns = ['Store', 'DayOfWeek', 'Promo']
df_means = train.groupby(grouped_columns)['Sales', 'Customers'].median().reset_index()
df_means['SalesMedian'] = df_means['Sales']
df_means['CustomerMedian'] = df_means['Customers']
df_means.drop(['Sales', 'Customers'], axis=1, inplace=True)
# print(df_means)

# grouped = train.groupby(['Store'], as_index=False)
# df_customer = grouped['Customers'].agg({'CustomersStoreMean': np.mean, 'CustomersStoreStd': np.std})
# df_sales = grouped['Sales'].agg({'SalesStoreMean': np.mean, 'SalesStoreStd': np.std})
# df_customer = train.merge(df_sales, on='Store', how='left')

# df_customer = train[['Store', 'Customers']].groupby(['Store']).mean().reset_index()
train.drop(['Customers'], axis=1, inplace=True)
train = train.merge(df_means, on=grouped_columns, how='left')
# print(train.head())
train = train.drop(['Store'], axis=1).astype(float)

features = train.columns.values.tolist()
features.remove('Sales')
# features.remove('Assortmentb')
# features.remove('StoreTypeb')
print(features)

test = build_features(test)
test = test.merge(df_means, on=grouped_columns, how='left')
print("any null test: ", np.unique(test[test.isnull().any(axis=1)][['DayOfWeek']].values))

# on sunday there is no sales data, we can safely replace NAN with 0
test.fillna(0, inplace=True)

test = test.drop(['Store'], axis=1).astype(float)

print('training data processed')

params = {"objective": "reg:linear",
          "eta": 0.3,
          "max_depth": 10,
          "subsample": 0.7,
          "colsample_bytree": 0.7,
          "silent": 1,
          "seed": 1301
          }
num_boost_round = 300

print("Train a XGBoost model")
X_train, X_valid = train_test_split(train, test_size=0.012)
y_train = np.log1p(X_train.Sales)
y_valid = np.log1p(X_valid.Sales)
dtrain = xgb.DMatrix(X_train[features], y_train)
dvalid = xgb.DMatrix(X_valid[features], y_valid)

watchlist = [(dvalid, 'eval'), (dtrain, 'train')]
gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist, early_stopping_rounds=100, \
                feval=rmspe_xg, verbose_eval=True)

print("Validating")
yhat = gbm.predict(xgb.DMatrix(X_valid[features]))
error = rmspe(X_valid.Sales.values, np.expm1(yhat))
print('RMSPE: {:.6f}'.format(error))

print("Make predictions on the test set")
dtest = xgb.DMatrix(test[features])
test_probs = gbm.predict(dtest)
# Make Submission
result = pd.DataFrame({"Id": test["Id"], 'Sales': np.expm1(test_probs)})
result.to_csv("xgboost_10_submission.csv", index=False)

# XGB feature importances
# Based on https://www.kaggle.com/mmueller/liberty-mutual-group-property-inspection-prediction/xgb-feature-importance-python/code

ceate_feature_map(features)
importance = gbm.get_fscore(fmap='xgb.fmap')
importance = sorted(importance.items(), key=operator.itemgetter(1))

df = pd.DataFrame(importance, columns=['feature', 'fscore'])
df['fscore'] = df['fscore'] / df['fscore'].sum()

featp = df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(6, 10))
plt.title('XGBoost Feature Importance')
plt.xlabel('relative importance')
fig_featp = featp.get_figure()
fig_featp.savefig('feature_importance_xgb.png', bbox_inches='tight', pad_inches=1)
