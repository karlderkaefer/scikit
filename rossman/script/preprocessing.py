from sklearn.decomposition import PCA
from sklearn.grid_search import GridSearchCV
from sklearn.tree import DecisionTreeRegressor

__author__ = 'Karl'
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import SGDRegressor
from time import process_time
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import *
from sklearn.svm import SVR, SVC
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
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


def merge(df):
    store = pd.read_csv('../input/store.csv')
    df = df.merge(store, on='Store', how="left")
    # Some Store have empty open value on test set
    df.loc[df.Open.isnull(), 'Open'] = 0
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
    return df


pd.set_option('display.width', 2800)




def submission(clf, X, y, test):
    t = process_time()
    df_test = merge(test)
    df_test = df_test.sort_index(axis=1).set_index('Id')
    X_test = df_test.fillna(0).astype(float).values
    X_test = StandardScaler().fit_transform(X_test)
    clf.fit(X, y)
    y_test = clf.predict(X_test)
    result = pd.DataFrame({'Id': df_test.index.values, 'Sales': y_test}).set_index('Id')
    result = result.sort_index()
    result.to_csv('submission.csv')
    result.to_csv('submission.excel.csv', sep=";", decimal=",")
    print('submission created in ', process_time() - t)

def submission_tree(clf, X, y, X_test_submission):
    t = process_time()
    clf.fit(X, y)
    y_predict = clf.predict(X_test_submission)
    y_predict = y_predict * X_test_submission['SalesStoreMean']
    result = pd.DataFrame({'Id': X_test_submission.index.values, 'Sales': y_predict}).set_index('Id')
    result = result.sort_index()
    result.to_csv('submission5.csv')
    result.to_csv('submission5.excel.csv', sep=";", decimal=",")
    print('submission created in ', process_time() - t)

def train_and_evaluate(name, clf, X_train, y_train):
    t = process_time()
    print("------------- ", name, " --------------")
    clf.fit(X_train, y_train)
    print(clf.score(X_train, y_train), " = Coefficient of determination on training set")
    cv = KFold(X_train.shape[0], 5, shuffle=True)
    scores = cross_val_score(clf, X_train, y_train, cv=cv)
    print("Cross validation accuracy: %.3f +/- %.3f" % (np.mean(scores), np.std(scores)))
    print('train and evaluate model in ', process_time() - t)

def score(name, clf, X_train, y_train, X_test, y_test):
    t = process_time()
    print("------------- ", name, " --------------")
    clf.fit(X_train, y_train)
    y_predict = clf.predict(X_test)
    print("RMSPE: %.6f" % rmspe(y_predict, y_test))
    # print("feature importance: " + np.array_str(clf.feature_importances_))
    print('train and evaluate model in ', process_time() - t)


def grid_parameter_tuning(name, clf, X_train, y_train):
    t = process_time()
    print("------------- Parameter Tuning PCA --------------")
    param_alpha = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    param_grid = dict(alpha=param_alpha)
    gs = GridSearchCV(clf, param_grid)
    gs = gs.fit(X_train, y_train)
    print(gs.grid_scores_)
    print(gs.best_params_)
    print('parameter tuning in %.3f' % (process_time() - t))


def estimate():
    # Load data
    t = process_time()
    data = pd.read_csv('../input/train.csv', parse_dates=['Date'], dtype={'StateHoliday': str})
    test = pd.read_csv('../input/test.csv', parse_dates=['Date'], dtype={'StateHoliday': str})

    print('training data loaded in ', process_time() - t)

    # exclude closed store
    data = data[data['Open'] != 0]

    # data = data[data['Store'] == 1]
    # test = data[data['Store'] == 1]
    print(data.head())

    # merge
    t = process_time()
    data = merge(data)
    print('training data processed in ', process_time() - t)

    # calculate avg of customer for each store
    df_customer = data[['Store', 'Customers']].groupby(['Store']).mean().reset_index()

    # grouped = data.groupby(['Store'], as_index=False)
    # df_customer = grouped['Customers'].agg({'CustomersStoreMean': np.mean, 'CustomersStoreStd': np.std})
    # df_sales = grouped['Sales'].agg({'SalesStoreMean': np.mean, 'SalesStoreStd': np.std})
    # df_customer = df_customer.merge(df_sales, on='Store')

    data = data.drop(['Customers'], axis=1)
    data = data.merge(df_customer, on='Store')
    data = data.drop(['Store'], axis=1)

    # Make sure columns are in right order
    X = data.sort_index(axis=1).fillna(0)
    X = X.drop(['Sales'], axis=1).astype(float)
    # y = data['Sales'] / data['SalesStoreMean']
    # print(y)
    y = data['Sales'].astype(float).values
    # print("features: " + X.columns.values)

    # Prepare Test Set
    t = process_time()
    test = merge(test)
    test = test.merge(df_customer, on='Store')
    test = test.drop(['Store'], axis=1).set_index('Id')
    X_test_submission = test.sort_index(axis=1).fillna(0).astype(float)
    print('test data processed in ', process_time() - t)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.70, random_state=2)
    # scaler = StandardScaler()
    # X_train_transformed = scaler.fit_transform(X_train)
    # X_test_transformed = scaler.fit_transform(X_test)

    clf = make_pipeline(StandardScaler(), SGDRegressor(loss='squared_loss', penalty='l2'))
    # grid_parameter_tuning("SGD", clf, X_train_transformed, y_train)
    # score("SDGRegressor", clf, X_train, y_train, X_test, y_test)
    train_and_evaluate("SDGRegressor", clf, X_train, y_train)
    # parameter_tuning("SDGRegressor", clf, X_train_transformed, y_train)

    clf = RandomForestRegressor(n_jobs=-1, n_estimators=25)
    # score("RandomForestRegressor", clf, X_train, y_train, X_test, y_test)
    train_and_evaluate("RandomForestRegressor", clf, X_train, y_train)
    # submission_tree(clf, X, y, X_test_submission)

    clf = DecisionTreeRegressor()
    train_and_evaluate("DecisionTreeRegressor", clf, X_train, y_train)


    #clf = GradientBoostingRegressor()
    #score("GradientBoosting", clf, X_train_transformed, y_train, X_test_transformed, y_test)


# def predict_customers():


estimate()




