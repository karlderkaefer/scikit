import pandas as pd
from sklearn.preprocessing import *
from sklearn.feature_extraction import DictVectorizer

store = pd.read_csv('../input/store.csv').fillna(0)
store = store[['Assortment', 'StoreType']]
print(store.head(10))

# enc = LabelBinarizer()
# store['Assortment'] = enc.fit_transform(store['Assortment'])
# print(store[store['Assortment'] == 1])

enc = DictVectorizer()
transformed = enc.fit_transform(store)
print(transformed)


# store_dict = store.T.to_dict().values()
# vec = DictVectorizer(sparse=False)
# store_vec = vec.fit_transform(store)
# print(store_vec.head())

