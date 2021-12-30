import os
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

path = os.path.dirname(__file__)
# print(path)
df = pd.read_csv(path + '/input/winequality_red.csv', sep=';', quotechar='"')


# print(df.head())
# df.info()
# print(df.describe())
# print(df['quality'].unique())
# print(df['qaulity'].value_counts())

# sns.countplot(x='quality', data=df)
# plt.show()

# fig=plt.figure(figsize=(10,6))
# sns.barplot(x='quality', y='volatile acidity', data=df)
# plt.show()

def isTasty(quality):
    return 1 if quality >= 6.5 else 0


df['tasty'] = df['quality'].apply(isTasty)
# print(df.columns)
# print(df.head())

# print(df['tasty'].value_counts())

X = df.drop(['quality', 'tasty'], axis=1)
y = df['tasty']

from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33, random_state=123)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
x_val = sc.fit_transform(X_val)

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=300)
rfc.fit(X_train, y_train)

y_pred = rfc.predict(X_val)

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
print(confusion_matrix(y_val, y_pred))
print(accuracy_score(y_val,y_pred))
print(classification_report(y_val, y_pred))