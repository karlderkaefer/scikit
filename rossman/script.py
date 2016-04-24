__author__ = 'Karl'

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn_pandas import DataFrameMapper, cross_val_score
from itertools import cycle
import pylab as pl
from sklearn import svm

from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
import sklearn.pipeline


def decision(data, target):
    model = DecisionTreeClassifier()
    model.fit(data[:-10], target[:-10])
    print(model)
    # make predictions
    expected = target[-10:]
    predicted = model.predict(data[-10:])
    print(predicted.shape)
    # summarize the fit of the model
    print(metrics.classification_report(expected, predicted))
    print(metrics.confusion_matrix(expected, predicted))


def cross_validation(df, mapper):
    pipe = sklearn.pipeline.Pipeline([
        ('featurized', mapper),
        ('lm', sklearn.linear_model.LinearRegression())
    ])
    result = cross_val_score(pipe, df.copy(), df.Sales)
    print(np.round(result), 2)


def write_merged_table():
    df_train = pd.read_csv("data/train.csv", dtype={'StateHoliday': str, 'Promo': bool, 'Open': bool})

    # filter columns
    df_train = df_train[['Store', 'Customers', 'Promo', 'Sales', 'Open']]

    # filter closed stores
    df_train = df_train[df_train['Open'] != 0]

    # show columns with missing values
    # print(df_train.count())

    df_store = pd.read_csv("data/store.csv", dtype={'Promo2': bool})
    df_store = df_store[['Store', 'CompetitionDistance', 'Promo2']]

    # show columns with missing values
    # print(df_store.count())

    df_merged = pd.merge(df_train, df_store, how="left", on=['Store', 'Store'])
     # fill null values
    df_merged = df_train.fillna(df_merged.mean())

    # print data types
    # print(df_merged.dtypes)
    print(df_merged.columns)

    # save table
    df_merged.to_csv("data/all.csv")

def full():
    df = pd.read_csv("data/train.csv", dtype={'StateHoliday': str})
    mapper_data = DataFrameMapper([
        # (['Store'], preprocessing.LabelBinarizer()),
        (['Customers'], preprocessing.StandardScaler())
    ])
    np_target = np.asarray(df['Sales'])
    np_data = mapper_data.fit_transform(df.copy())

    # reduce compution time
    np_target = np_target[:10000]
    np_data = np_data[:10000]

    X_train, X_test, y_train, y_test = sklearn.cross_validation.train_test_split(np_data, np_target, test_size=0.1,
                                                                                 random_state=0)
    print("train size: ", X_train.shape)
    print("test size: ", X_test.shape)

    print("train data: ", X_train[:2])
    print("target data: ", y_train[:2])

    # clf = sklearn.linear_model.SGDClassifier(alpha=0.001, n_iter=20).fit(X_train, y_train)
    clf = sklearn.svm.SVC(C=1).fit(X_train, y_train)
    print(clf.score(X_test, y_test))



def estimator_sgd():
    # mapper_data = DataFrameMapper([
    #     (['Store'], preprocessing.L)
    #     (['Customers'], preprocessing.StandardScaler())
    # ])
    df = pd.read_csv("data/all.csv", dtype={'Customers': np.float})
    # np_target = np.asarray(df['Sales'])
    # np_data = mapper_data.fit_transform(df)
    np_data = np.asarray(df[['Customers', 'Store']])
    np_target = np.asarray(df['Sales'])

    # reduce compution time
    np_target = np_target[:10000]
    np_data = np_data[:10000]

    # print(np_data)
    # print(np_target)

    X_train, X_test, y_train, y_test = sklearn.cross_validation.train_test_split(np_data, np_target, test_size=0.2,
                                                                                 random_state=0)

    print("train size: ", X_train.shape)
    print("test size: ", X_test.shape)

    scaler = preprocessing.StandardScaler()
    scaler.fit(X_train)
    X_train_transformed = scaler.transform(X_train[[0]])
    X_test_transformed = scaler.transform(X_test[[0]])


    print("transformed: ", X_train_transformed)

    clf = sklearn.linear_model.SGDClassifier(alpha=0.001, n_iter=10).fit(X_train_transformed, y_train)
    # clf = sklearn.svm.SVC(C=1).fit(X_train_transformed, y_train)
    print(clf.score(X_test_transformed, y_test))


# write_merged_table()
# estimator_sgd()
full()


# print(np_data)
# print(np_target)
# print("size: ", np_data.shape)
# print("size: ", np_target.shape)
# print("size: ", df_merged.shape)



# decision(np_data[:1000], np_target[:1000])
# cross_validation(df_merged.head(10000), mapper_data)






# clf = svm.SVC(gamma=0.001, C=100)
# x_train, y_train = np_data[:-100000], np_target[:-100000]

# clf.fit(x_train, y_train)
# print(clf.score)



# df_target = df_merged['Sales']
# np_data = np.asarray(df_merged)
# np_target = np.asarray(df_target)
#
# print("targets", np_target)
# print("data", np_data)
#
# # gamma = learning rate
# clf = svm.SVC(gamma=0.001, C=100)
# x, y = np_data[:-100], np_target[:-100]
# clf.fit(x, y)
#
# print('Prediction:', clf.predict(np_data[-10]))
#
#
# def print_diagram():
#     df_merged["CompetitionDistance"][np.isnan(["CompetitionDistance"])] = np.median(df_merged["CompetionDistance"])
#
#     #train = pd.read_csv("data/all.csv", dtype={'StateHoliday': str})
#
#     plt.figure()
#     sns.pairplot(data=df_merged[["Sales", "CompetitionDistance", "Customers"]], dropna=True)
#     plt.savefig("data/plot.png")
