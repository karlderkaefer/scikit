from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cross_validation import train_test_split, StratifiedKFold

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data', header=None)

X = df.loc[:, 2:]
y = df.loc[:, 1]
le = LabelEncoder()
y = le.fit_transform(y)
print(le.transform(['M', 'B']))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

pipe = Pipeline([
    ('scl', StandardScaler()),
    ('pca', PCA(n_components=2)),
    ('clf', LogisticRegression(random_state=1))
])

pipe.fit(X_train, y_train)
print("Test Accuracy: %.3f" % pipe.score(X_test, y_test))

kfold = StratifiedKFold(y_train, n_folds=10, random_state=1)
scores = []
for k, (train, test) in enumerate(kfold):
    print(train)
    pipe.fit(X_train[train], y_train[train])
    score = pipe.score(X_train[test], y_train[test])
    scores.append(score)
    print("Fold: %s, Class dist.: %s, Acc: %.3f" % (k+1, np.bincount(y_train[train]), score))

print("CV accuracy: %.3f +/- %.3f" % (np.mean(scores), np.std((scores))))

