import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

df = pd.read_csv('./titanic.csv')
df['male'] = df['Sex'] == 'male'
X = df[['Pclass', 'male', 'Age', 'Siblings/Spouses','Parents/Children']]
y = df['Survived'].values

model = LogisticRegression()
model.fit(X,y)
y_pred = model.predict(X)

print('accuracy : ', accuracy_score(y, y_pred))
print('precision : ', precision_score(y, y_pred))
print('recall : ', recall_score(y, y_pred))
print('f1 score : ', f1_score(y, y_pred))

print(confusion_matrix(y, y_pred))
