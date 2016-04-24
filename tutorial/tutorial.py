import pandas as pd

# df is type if pandas DataFrame
df = pd.read_csv('input/train.csv')
# print first 5 rows
print(df.head())
# print some statistics for each column
print(df.describe())

# select one column
df1 = df['temp']
print(df1.head())
# select two columns
df2 = df[['temp', 'weather']]
print(df2.head())
# add column to DataFrame
df['newcolumn'] = 2
# delete column
del df['season']
del df['datetime']
# filter column by condition
df3 = df[(df['weather'] > 3)]
print(df3.head())
# compute with columns
df['t-at'] = df['temp'] - df['atemp']
print(df.head())
# calculate difference to previous row
df['Difference'] = df['temp'].diff()
print(df['Difference'].head())
# calculate mean of nearest 100 rows
df['Average100'] = pd.rolling_mean(df['temp'], 100)
print(df['Average100'].head())

df = pd.read_csv('input/train.csv')
print(df.dtypes)

df = pd.read_csv('input/train.csv')
df['datetime'] = pd.to_datetime(df['datetime'], format='%Y-%m-%d %H:%M:%S')
print("convert to datetime after read csv: ", df.dtypes)

custom_parser = lambda date: pd.datetime.strptime(date, '%Y-%m-%d %H:%M:%S')
df = pd.read_csv('input/train.csv', parse_dates=['datetime'], date_parser=custom_parser)
print("use custom parser: ", df.dtypes)

df = pd.read_csv('input/train.csv', parse_dates=['datetime'])
print(df.dtypes)

import pylab as plt
import seaborn as sns

plt.figure()
sns.set(style="ticks", color_codes=True)
df_summer = df[(df['season'] == 2)]
sns.pairplot(df_summer[:100],
             x_vars=["windspeed", "humidity", "temp"],
             y_vars="count",
             kind='reg', size=4, aspect=0.6)
plt.savefig("plot.png")

df['year'] = df.datetime.apply(lambda x: x.year)
df['month'] = df.datetime.apply(lambda x: x.month)
df['dow'] = df.datetime.apply(lambda x: x.dayofweek)
df['woy'] = df.datetime.apply(lambda x: x.weekofyear)
df['hour'] = df.datetime.apply(lambda x: x.hour)

df = pd.get_dummies(df, columns=['season', 'dow'])

# we wont use causal and registered in tutorial
df.drop(['registered', 'casual'], axis=1, inplace=True)
df.drop(['datetime'], axis=1, inplace=True)

from sklearn.cross_validation import train_test_split
# X are our features without 'count'
X = df.drop(['count'], axis=1)
# y is the target 'count'
y = df['count']
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.70, random_state=2)

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor

X_train_transformed = StandardScaler().fit_transform(X_train)
X_test_transformed = StandardScaler().fit_transform(X_test)
clf = SGDRegressor()
# train you model
clf.fit(X_train_transformed, y_train)
# print out predicted values
print(clf.predict(X_test_transformed))
# print out how well the model fits for data
print("Model Score: ", clf.score(X_train_transformed, y_train))

from sklearn.pipeline import make_pipeline
clf = make_pipeline(StandardScaler(), SGDRegressor())
clf.fit(X_train, y_train)
print("Model Score: ", clf.score(X_train, y_train))
