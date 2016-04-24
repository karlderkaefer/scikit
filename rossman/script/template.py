import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import train_test_split


# Thanks to chenglongchen RMSE calculation script

def ToWeight(y):
    w = np.zeros(y.shape, dtype=float)
    ind = y != 0
    w[ind] = 1. / (y[ind] ** 2)
    return w


def RMSPE(y, yhat):
    w = ToWeight(y)
    return np.sqrt(np.mean(w * (y - yhat) ** 2))



def process_data(data):
    # Merge store data
    data = data.merge(store, on='Store', copy=False)

    # Break down date column
    data['year'] = data.Date.apply(lambda x: x.year)
    data['month'] = data.Date.apply(lambda x: x.month)
    #     data['dow'] = data.Date.apply(lambda x: x.dayofweek)
    data['woy'] = data.Date.apply(lambda x: x.weekofyear)
    data.drop(['Date'], axis=1, inplace=True)

    # Calculate time competition open time in months
    data['CompetitionOpen'] = 12 * (data.year - data.CompetitionOpenSinceYear) + \
                              (data.month - data.CompetitionOpenSinceMonth)
    data['CompetitionOpen'] = data.CompetitionOpen.apply(lambda x: x if x > 0 else 0)
    data.drop(['CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear'], axis=1,
              inplace=True)

    # Promo open time in months
    data['PromoOpen'] = 12 * (data.year - data.Promo2SinceYear) + \
                        (data.woy - data.Promo2SinceWeek) / float(4)
    data['PromoOpen'] = data.CompetitionOpen.apply(lambda x: x if x > 0 else 0)
    data.drop(['Promo2SinceYear', 'Promo2SinceWeek'], axis=1,
              inplace=True)

    # Get promo months
    data['p_1'] = data.PromoInterval.apply(lambda x: x[:3] if type(x) == str else 0)
    data['p_2'] = data.PromoInterval.apply(lambda x: x[4:7] if type(x) == str else 0)
    data['p_3'] = data.PromoInterval.apply(lambda x: x[8:11] if type(x) == str else 0)
    data['p_4'] = data.PromoInterval.apply(lambda x: x[12:15] if type(x) == str else 0)

    print(data[data["p_1"] != 0])


    # Get dummies for categoricals
    data = pd.get_dummies(data, columns=['p_1', 'p_2', 'p_3', 'p_4',
                                         'StateHoliday',
                                         'StoreType',
                                         'Assortment'])
    data.drop(['Store',
               'PromoInterval',
               'p_1_0', 'p_2_0', 'p_3_0', 'p_4_0',
               'StateHoliday_0',
               'year'], axis=1, inplace=True)


    # Fill in missing values
    data = data.fillna(0)
    data = data.sort_index(axis=1)
    return data


## Start of main script

# Load data
data = pd.read_csv('../input/train.csv', parse_dates=['Date'])
store = pd.read_csv('../input/store.csv')
print('training data loaded')

# Only use stores that are open to train
data = data[data['Open'] != 0]

# Process training data
data = process_data(data)
print('training data processed')

# Set up training data
X_train = data.drop(['Sales', 'Customers'], axis=1)
y_train = data.Sales
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.03, random_state=0)

# Fit random forest model
model = RandomForestRegressor(n_jobs=-1, n_estimators=15)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
error = RMSPE(y_test, y_pred)
print("RMSPE:", error)
print('model fit')

# Load and process test data
test = pd.read_csv('../input/test.csv', parse_dates=['Date'])
test = process_data(test)

# Ensure same columns in test data as training
for col in data.columns:
    if col not in test.columns:
        test[col] = np.zeros(test.shape[0])

test = test.sort_index(axis=1).set_index('Id')
print('test data loaded and processed')

# Make predictions
X_test = test.drop(['Sales', 'Customers'], axis=1).values
y_test = model.predict(X_test)

# Make Submission
result = pd.DataFrame({'Id': test.index.values, 'Sales': y_test}).set_index('Id')
result = result.sort_index()
result.to_csv('submission.csv')
print('submission created')
