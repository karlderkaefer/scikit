__author__ = 'Karl'

import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import SGDRegressor
from time import process_time
import matplotlib.pyplot as plt

# Thanks to Chenglong Chen for providing this in the forum
def to_weight(y):
    w = np.zeros(y.shape, dtype=float)
    ind = y != 0
    w[ind] = 1. / (y[ind] ** 2)
    return w


def rmspe(yhat, y):
    w = to_weight(y)
    return np.sqrt(np.mean(w * (y - yhat) ** 2))

def sparsity_ratio(X):
    return 1.0 - np.count_nonzero(X) / float(X.shape[0] * X.shape[1])

def process_traindata(df):
    # Merge store data
    df = df.merge(store, on='Store', how="left")

    # Break down date column
    df['year'] = df.Date.apply(lambda x: x.year)
    df['month'] = df.Date.apply(lambda x: x.month)
    # data['dow'] = data.Date.apply(lambda x: x.dayofweek)
    df['woy'] = df.Date.apply(lambda x: x.weekofyear)
    df.drop(['Date'], axis=1, inplace=True)
    df.drop(['Open'], axis=1, inplace=True)


    # Calculate time competition open time in months
    df['CompetitionOpen'] = 12 * (df.year - df.CompetitionOpenSinceYear) + \
                              (df.month - df.CompetitionOpenSinceMonth)
    df['CompetitionOpen'] = df.CompetitionOpen.apply(lambda x: x if x > 0 else 0)
    df.drop(['CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear'], axis=1,
              inplace=True)

    # Promo open time in months
    df['PromoOpen'] = 12 * (df.year - df.Promo2SinceYear) + \
                        (df.woy - df.Promo2SinceWeek) / float(4)
    df['PromoOpen'] = df.PromoOpen.apply(lambda x: x if x > 0 else 0)
    df.drop(['Promo2SinceYear', 'Promo2SinceWeek'], axis=1,
              inplace=True)

    # Get promo months
    df['prom'] = df.PromoInterval.apply(lambda x: x[0] if type(x) == str else 'N')
    # data['p_1'] = data.PromoInterval.apply(lambda x: 1 if (type(x) == str and x.startswith('Jan')) else 0)
    # data['p_2'] = data.PromoInterval.apply(lambda x: 2 if (type(x) == str and x.startswith('Feb')) else 0)
    # data['p_3'] = data.PromoInterval.apply(lambda x: 3 if (type(x) == str and x.startswith('Mar')) else 0)
    # data['p_4'] = data.PromoInterval.apply(lambda x: 4 if (type(x) == str and x.startswith('Jan')) else 0)


    # Get dummies for categoricals
    df = pd.get_dummies(df, columns=['prom',
                                         'StateHoliday',
                                         'StoreType',
                                         'Assortment'
                                        ])
    df.drop('StateHoliday_0', axis=1, inplace=True)

    df.drop(['Store',
               'PromoInterval',
               'year'], axis=1, inplace=True)

    # data.to_csv("test.csv", sep=";", decimal=",")

    # Fill in missing values
    # data = data.fillna(0)
    df = df.fillna(0)
    # data = data.sort_index(axis=1)

    # data.to_csv("test2.csv", sep=";", decimal=",")
    df.sort_index(axis=1)

    return df


def submission(df_train):
    df_test = pd.read_csv('../input/test.csv', parse_dates=['Date'], dtype={'StateHoliday': str})
    df_test = process_traindata(df_test)

    # Ensure same columns in test data as training
    for col in df_train.columns:
        if col not in df_test.columns:
            df_test[col] = np.zeros(df_test.shape[0])

    df_test = df_test.sort_index(axis=1).set_index('Id')
    df_train = df_train.sort_index(axis=1)
    print('target: ', df_train_target)
    clf_sdg = SGDRegressor()
    clf_sdg.fit(df_train, df_train_target)
    y_pred = clf_sdg.predict(df_test)
    print('predicted: ', np.rint(y_pred))

    # Make Submission
    result = pd.DataFrame({'Id': df_test.index.values, 'Sales': y_pred}).set_index('Id')
    result = result.sort_index()
    result.to_csv('submission.csv')
    result.to_csv('submission.excel.csv', sep=";", decimal=",")
    print('submission created')


pd.set_option('display.width', 400)

t = process_time()
# Load data
data = pd.read_csv('../input/train.csv', parse_dates=['Date'], dtype={'StateHoliday': str})
store = pd.read_csv('../input/store.csv')
print('training data loaded in ', process_time() - t)


# exclude closed store
data = data[data['Open'] != 0]

t = process_time()
data = process_traindata(data)
print('training data processed in ', process_time() - t)
t = process_time()
# Set up training data

df_train = data.drop(['Sales', 'Customers'], axis=1)
df_train_target = data.Sales
X_train, X_test, y_train, y_test = train_test_split(df_train, df_train_target, test_size=0.02, random_state=0)

print("input sparsity ratio:", sparsity_ratio(X_train))
model = SGDRegressor()
clf = model.fit(X_train, y_train)
print(clf.coef_)
print('model fit processed in ', process_time() - t)
print(np.round(clf.score(X_test, y_test)), 2)
y_pred = clf.predict(X_test)
error = rmspe(y_test, y_pred)
print("RMSPE:", error)

# predict for submission
submission(df_train)


# # Plot feature importance
# feature_importance = clf.coef_
# # make importances relative to max importance
# feature_importance = 100.0 * (feature_importance / feature_importance.max())
# sorted_idx = np.argsort(feature_importance)
# pos = np.arange(sorted_idx.shape[0]) + .5
# plt.figure()
# plt.barh(pos, feature_importance[sorted_idx], align='center')
# plt.yticks(pos, X_train.columns.values[sorted_idx])
# plt.xlabel('Relative Importance')
# plt.title('Variable Importance')
# plt.show()

