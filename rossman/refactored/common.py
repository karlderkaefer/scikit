from time import process_time
from sklearn.cross_validation import cross_val_score, KFold, \
    cross_val_predict
from sklearn.ensemble.partial_dependence import \
    plot_partial_dependence
import numpy as np
import pandas as pd
import pylab as plt
from operator import itemgetter
import seaborn as sns

import warnings

warnings.filterwarnings("ignore")


def to_weight(y):
    w = np.zeros(y.shape, dtype=float)
    ind = y != 0
    w[ind] = 1. / (y[ind] ** 2)
    return w


# Thanks to Chenglong Chen for providing this in the forum
def rmspe(y, yhat):
    w = to_weight(y)
    return np.sqrt(np.mean(w * (y - yhat) ** 2))


def rmspe_xg(yhat, y):
    y = np.expm1(y.get_label())
    yhat = np.expm1(yhat)
    return "rmspe", rmspe(y, yhat)


# def rmspe(y_true, y_pred):
#     sum = 0.0
#     for y, yhat in zip(y_true, y_pred):
#         sum += np.square((y - yhat) / y)
#     return np.sqrt(sum / y_true.size)


# feature extraction, building categories and merging dataframes
def build_features(df, store):
    # store = pd.read_csv('../input/store.csv')
    df = df.merge(store, on='Store', how="left")
    # Break down date column
    df['year'] = df.Date.apply(lambda x: x.year)
    df['month'] = df.Date.apply(lambda x: x.month)
    df['woy'] = df.Date.apply(lambda x: x.weekofyear)

    # Calculate time competition open time in months
    df['CompetitionOpen'] = 12 * (
    df.year - df.CompetitionOpenSinceYear) + (
                            df.month - df.CompetitionOpenSinceMonth)
    df['CompetitionOpen'] = df.CompetitionOpen.apply(
        lambda x: x if x > 0 else 0)

    # Promo open time in months
    df['PromoOpen'] = 12 * (df.year - df.Promo2SinceYear) + \
                      (df.woy - df.Promo2SinceWeek) / float(4)
    df['PromoOpen'] = df.PromoOpen.apply(lambda x: x if x > 0 else 0)
    df['PromoInterval'] = df.PromoInterval.apply(
        lambda x: x[0] if type(x) == str else 'N')

    df['StateHoliday'] = df.StateHoliday.apply(
        lambda x: 1 if (x != '0') else 0)

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
    # for xgb
    df = df.rename(columns=lambda x: x.replace('_', ''))
    return df


# retrieve only train set
def get_trainset():
    t = process_time()
    train = pd.read_csv('../input/train.csv', parse_dates=['Date'],
                        dtype={'StateHoliday': str})
    store = pd.read_csv('../input/store.csv')
    print('reading data processed in ', process_time() - t)
    # filter unuseful data
    train = train[train["Open"] != 0]
    train = train[train["Sales"] > 0]
    train = build_features(train, store)

    # calculate customer and sales median for each store
    grouped_columns = ['Store', 'DayOfWeek', 'Promo']
    means = train.groupby(grouped_columns)[
        'Sales', 'Customers'].median().reset_index()
    means['SalesMedian'] = means['Sales']
    means['CustomerMedian'] = means['Customers']
    means.drop(['Sales', 'Customers'], axis=1, inplace=True)

    # merge median with train set
    train.drop(['Customers'], axis=1, inplace=True)
    train = train.merge(means, on=grouped_columns, how='left')
    train = train.drop(['Store'], axis=1).astype(float)

    # on sunday there is no sales data, we can safely replace NAN with 0
    # np.unique(test[test.isnull().any(axis=1)][['DayOfWeek']].values)
    train.fillna(0, inplace=True)

    feature_names = train.columns.values.tolist()
    feature_names.remove('Sales')

    print('merge train and test data and feature extraction in ',
          process_time() - t)

    return train, feature_names


# return train and test set and features names
# pass store_id (List) to use only specified stores
# feature extraction: Sales and Customer for each Store
def get_sets(store_id=None):
    t = process_time()
    train = pd.read_csv('../input/train.csv', parse_dates=['Date'],
                        dtype={'StateHoliday': str})
    test = pd.read_csv('../input/test.csv', parse_dates=['Date'],
                       dtype={'StateHoliday': str})
    store = pd.read_csv('../input/store.csv')
    print('reading data processed in ', process_time() - t)

    if store_id is not None:
        print("found store id. only process model for store ",
              store_id)
        train = train[train['Store'] == store_id]
        test = test[test['Store'] == store_id]

    train = train[train["Open"] != 0]
    train = train[train["Sales"] > 0]
    # NaN in test should be handled as open stores
    test[['Open']] = test[['Open']].fillna(1)

    train = build_features(train, store)
    test = build_features(test, store)
    print('merging store processed in ', process_time() - t)

    # calculate customer and sales median for each store
    grouped_columns = ['Store', 'DayOfWeek', 'Promo']
    means = train.groupby(grouped_columns)['Sales', 'Customers'] \
        .median().reset_index()
    means['SalesMedian'] = means['Sales']
    means['CustomerMedian'] = means['Customers']
    means.drop(['Sales', 'Customers'], axis=1, inplace=True)

    # merge median with train set
    train.drop(['Customers'], axis=1, inplace=True)
    train = train.merge(means, on=grouped_columns, how='left')
    test = test.merge(means, on=grouped_columns, how='left')
    train = train.drop(['Store'], axis=1).astype(float)
    test = test.drop(['Store'], axis=1).astype(float)

    feature_names = train.columns.values.tolist()
    feature_names.remove('Sales')

    # on sunday there is no sales data,
    # we can safely replace NAN with 0
    # print("any null test: ", np.unique
    # (test[test.isnull().any(axis=1)][['DayOfWeek']].values))
    test.fillna(0, inplace=True)
    train.fillna(0, inplace=True)

    print('merge train and test data and feature extraction in ',
          process_time() - t)

    return train, test, feature_names


# try to combine google searches for "rossman" with dataset
# actually this is nonsense,
# because you cant use this data in test set
def external_store(df, store):
    states = pd.read_csv('../input/external/store_states.csv')
    state_name = np.unique(states['State'])
    dateparse = lambda x: pd.datetime.strptime(
        x.split(" - ")[0], '%Y-%m-%d')
    print("adding external for states ", state_name)
    googlestats = pd.DataFrame(index=['IdGoogle'], columns=[
        'rossmann', 'year', 'woy', 'Store'])
    print(googlestats.head())
    for i, stateitem in enumerate(state_name):
        path = "../input/external/Rossmann_DE_" + stateitem + ".csv"
        googlestats_state = pd.read_csv(path, parse_dates=['Week'],
                                        date_parser=dateparse)
        googlestats_state['year'] = googlestats_state.Week.apply(
            lambda x: x.year)
        googlestats_state['woy'] = googlestats_state.Week.apply(
            lambda x: x.weekofyear)
        googlestats_state.drop(['Week'], axis=1, inplace=True)
        googlestats_state['State'] = stateitem
        googlestats_state = googlestats_state.merge(states,
                                                    on='State',
                                                    how='left')
        googlestats_state.drop(['State'], axis=1, inplace=True)
        print(googlestats_state.head())
        googlestats = googlestats.append(googlestats_state,
                                         ignore_index=True)
        # print("append:", stateitem, googlestats.head())
    print(googlestats.shape)
    df = df.merge(store, on='Store', how="left")
    return df


def kfold_validation(name, clf, X_train, y_train):
    t = process_time()
    print("------------- ", name, " --------------")
    clf.fit(X_train, y_train)
    print(clf.score(X_train, y_train),
          " = Coefficient of determination on training set")
    cv = KFold(X_train.shape[0], 5, shuffle=True)
    scores = cross_val_score(clf, X_train, y_train, cv=cv)
    print("Cross validation accuracy: %.3f +/- %.3f" % (
    np.mean(scores), np.std(scores)))
    print('train and evaluate model in ', process_time() - t)


def train_score(name, clf, X_train, y_train):
    t = process_time()
    print("------------- ", name, " --------------")
    clf.fit(X_train, y_train)
    print("%s = Model Accuracy" % clf.score(X_train, y_train))
    print('calculate accuracy in ', process_time() - t)


def crossval(name, clf, X_train, y_train):
    t = process_time()
    print("------------- ", name, " --------------")
    clf.fit(X_train, y_train)
    scores = cross_val_score(clf, X_train, y_train)
    print("Cross validation accuracy: %.3f +/- %.3f" % (
    np.mean(scores), np.std(scores)))
    print('train and evaluate model in ', process_time() - t)


def score(name, clf, X_train, y_train, X_test, y_test):
    t = process_time()
    print("------------- ", name, " --------------")
    clf.fit(X_train, y_train)
    y_predict = clf.predict(X_test)
    print("RMSPE: %.6f" % rmspe(y_predict, y_test))
    # print("feature importance: " + np.array_str(clf.feature_importances_))
    print('train and evaluate model in ', process_time() - t)


def submission(clf, train, test, feature_names, label):
    t = process_time()
    clf.fit(train[feature_names], train.Sales.values)
    y_test = clf.predict(test[feature_names])
    result = pd.DataFrame({'Id': test.astype(int).Id.values,
                           'Sales': y_test}).set_index('Id')
    result = result.sort_index()
    result.to_csv("submission." + label + ".csv")
    result.to_csv("submission.excel." + label + ".csv", sep=";",
                  decimal=",")
    print('submission created in ', process_time() - t)


def print_cv_score(clf, X_train, y_train, label):
    scores = cross_val_score(clf, X_train, y_train,
                             scoring="mean_squared_error", cv=5)
    print("%.3f +/- %.3f = %s" % (
    np.mean(scores), np.std(scores), label))


def print_prediction_err(clf, X_train, y_train, label):
    prediction = cross_val_predict(clf, X_train, y_train)
    print("%.6f = RMSPE %s" % (
    rmspe(y_train.values, prediction), label))


# default feature names
def get_feature_names():
    feature_names = \
        ['DayOfWeek',
         'Promo',
         'StateHoliday',
         'SchoolHoliday',
         'CompetitionDistance',
         'Promo2',
         'month',
         'woy',
         'CompetitionOpen',
         'PromoOpen',
         'PromoIntervalF',
         'PromoIntervalJ',
         'PromoIntervalM',
         'PromoIntervalN',
         'StoreTypea',
         'StoreTypeb',
         'StoreTypec',
         'StoreTyped',
         'Assortmenta',
         'Assortmentb',
         'Assortmentc',
         'SalesMedian',
         'CustomerMedian']
    return feature_names


# Utility function to report best scores
# from http://scikit-learn.org/stable/auto_examples/model_selection/randomized_search.html
def report(grid_scores, n_top=3):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[
                 :n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
            score.mean_validation_score,
            np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")


def plot_regression_error(clf, X_train, y_train, name):
    predicted = cross_val_predict(clf, X_train, y_train, cv=5)
    sns.regplot(x=y_train, y=predicted)
    sns.axlabel("actual", "predicted")
    plt.savefig("plot_validation_" + name + ".png")


def plot_gradiant(clf, X_train, y_train, features, feature_name):
    clf.fit(X_train, y_train)
    fig, axs = plot_partial_dependence(clf, X_train, features,
                                       feature_names=feature_name,
                                       grid_resolution=50)
    fig.suptitle(
        'Partial dependence of house value on nonlocation features\n'
        'for the California housing dataset')
    plt.subplots_adjust(top=0.9)
    fig = plt.figure()
    plt.show()
