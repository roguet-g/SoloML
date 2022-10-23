from statistics import mode
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

df = pd.read_csv('./titanic.csv')
df['male'] = df['Sex'] == 'male'
X = df[['Pclass', 'male', 'Age', 'Siblings/Spouses','Parents/Children']]
y = df['Survived'].values

X_train, X_test, y_train, y_test = train_test_split(X,y)

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred_proba_1 = model.predict_proba(X_test)

print('model 1 AUC score', roc_auc_score(y_test, y_pred_proba_1[:,1]))

model2 = LogisticRegression()
model2.fit(X_train[:,0:2], y_train)
y_pred_proba_2 = model2.predict_proba(X_test[:,0:2])

print('model 2 AUC score ', roc_auc_score(y_test, y_pred_proba_2[:,1]))
