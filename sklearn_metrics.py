import pandas as pd
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('./titanic.csv')
df['male'] = df['Sex'] == 'male'
X = df[['Pclass', 'male', 'Age', 'Siblings/Spouses','Parents/Children']]
y = df['Survived'].values

model = LogisticRegression()
model.fit(X,y)
y_pred = model.predict(X)