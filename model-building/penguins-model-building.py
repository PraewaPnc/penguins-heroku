import pandas as pd
penguins = pd.read_csv('penguins_cleaned.csv')

# Ordinal feature encoding
# https://www.kaggle.com/pratik1120/penguin-dataset-eda-classification-and-clustering
df = penguins.copy()
target = 'species'
encode = ['sex', 'island']

for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df, dummy], axis=1)
    del df[col]

target_mapper = {'Adelie':0, 'Chinstrap':1, 'Gentoo':2}
def target_encode(key):
    return target_mapper[key]

df['species'] = df['species'].apply(target_encode)

# Separating X and y
X = df.drop('species', axis=1)
y = df['species']

# Build random forext model
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X, y)

# Saving the model
import pickle
pickle.dump(clf, open('penguins_clf.pkl', 'wb'))
