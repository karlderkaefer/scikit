from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from script import features as f

train, test, feature_names = f.get_sets(1)
dict = train[feature_names].to_dict()
print(dict)
X_train, X_test, y_train, y_test = train_test_split(train[feature_names], train.Sales.values, test_size=0.10, random_state=2)

clf = make_pipeline(StandardScaler(), SGDRegressor(loss='squared_loss', penalty='l2'))
f.score("SGDRegressor", clf, X_train, y_train, X_test, y_test)
f.kfold_validation("SGDRegressor", clf, X_train, y_train)
f.submission(clf, train, test, feature_names)



# clf = DecisionTreeRegressor()
# f.score("DecisionTreeRegressor", clf, X_train, y_train, X_test, y_test)
# f.kfold_validation("DecisionTreeRegressor", clf, X_train, y_train)
#
# clf = RandomForestRegressor(n_jobs=-1, n_estimators=25)
# f.score("RandomForestRegressor", clf, X_train, y_train, X_test, y_test)
# f.kfold_validation("RandomForestRegressor", clf, X_train, y_train)


# clf = GradientBoostingRegressor()
# f.score("GradientBoostingRegressor", clf, X_train, y_train, X_test, y_test)
# f.kfold_validation("GradientBoostingRegressor", clf, X_train, y_train)
