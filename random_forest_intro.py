import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

cancer_data = load_breast_cancer()
df = pd.DataFrame(cancer_data['data'], columns=cancer_data['feature_names'])
df['target'] = cancer_data['target']

X = df[cancer_data.feature_names].values
y = df['target'].values

print('Data dimension : ', X.shape)

X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=101)

rf = RandomForestClassifier()
rf.fit(X_train, y_train)

dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)

first_row = X_test[0]
print('predict : ', rf.predict([first_row]))
print('true value : ', y_test[0])

print('Random forest accuracy : ', rf.score(X_test, y_test))
print('Decision Tree accuracy : ', dt.score(X_test, y_test))