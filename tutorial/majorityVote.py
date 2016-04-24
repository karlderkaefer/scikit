from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.linear_model import SGDRegressor, Ridge, ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import common as c

def print_cv(labels, pipelines):
    for clf, label in zip(pipelines, labels):
        scores = cross_val_score(clf, X_train, y_train,
                                 scoring="r2");
        print("R2: %0.2f (+/- %0.2f [%s]" % (scores.mean(),
                                              scores.std(), label))

X, y = c.get_set()

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.5,
                                                    random_state=1)
pipe1 = Pipeline([['scl', StandardScaler()],
                  ['clf', SGDRegressor()]])
pipe2 = Pipeline([['scl', StandardScaler()],
                  ['clf', SGDRegressor()]])
pipe3 = Pipeline([['scl', StandardScaler()],
                  ['clf', SGDRegressor()]])
pipe4 = MajorityVoteR

clf_labels = ['SGDRegressor', 'Ridge', 'ElasticNet']
pipelines = [pipe1, pipe2, pipe3]

print_cv(clf_labels, pipelines)


