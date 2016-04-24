from sklearn.cross_validation import cross_val_predict, KFold
from sklearn.metrics import mean_squared_error
import pandas as pd
import pylab as plt
import seaborn as sns
import numpy as np


def get_set():
    df = pd.read_csv('input/train.csv', parse_dates=['datetime'])
    df['year'] = df.datetime.apply(lambda x: x.year)
    df['month'] = df.datetime.apply(lambda x: x.month)
    df['dow'] = df.datetime.apply(lambda x: x.dayofweek)
    df['woy'] = df.datetime.apply(lambda x: x.weekofyear)
    df['hour'] = df.datetime.apply(lambda x: x.hour)
    # df = pd.get_dummies(df, columns=['season', 'dow'])
    # we wont use causal and registered in tutorial
    df.drop(['registered', 'casual'], axis=1, inplace=True)
    df.drop(['datetime'], axis=1, inplace=True)
    X = df.drop(['count'], axis=1)
    y = df['count']
    return X, y


def plot_prediction_error(name, clf, X, y):
    plt.figure()
    cv = KFold(X.shape[0], 5, shuffle=True)
    predicted = cross_val_predict(clf, X, y, cv=cv)
    plot(name, predicted, y)
    print("%.3f = mean squared error" % mean_squared_error(y, predicted))


def plot(name, y_pred, y_test):
    print("%.3f = mean squared error" % mean_squared_error(y_test, y_pred))
    sns.regplot(x=y_test[:1000], y=y_pred[:1000])
    sns.axlabel("actual", "predicted")
    plt.savefig("plot_validation_" + name + ".png")

def get_full_set():
    df = pd.read_csv('input/train.csv', parse_dates=['datetime'])
    df['year'] = df.datetime.apply(lambda x: x.year)
    df['month'] = df.datetime.apply(lambda x: x.month)
    df['dow'] = df.datetime.apply(lambda x: x.dayofweek)
    df['woy'] = df.datetime.apply(lambda x: x.weekofyear)
    df['hour'] = df.datetime.apply(lambda x: x.hour)
    # df = pd.get_dummies(df, columns=['season', 'dow'])
    # we wont use causal and registered in tutorial
    df.drop(['registered', 'casual'], axis=1, inplace=True)
    #df.drop(['datetime'], axis=1, inplace=True)
    X = df.drop(['count'], axis=1)
    y = df['count']

    features = ['season', 'holiday', 'workingday',  'weather',
               'temp', 'humidity', 'windspeed', 'year',
               'month', 'dow', 'woy', 'hour']
    return X, y, features

