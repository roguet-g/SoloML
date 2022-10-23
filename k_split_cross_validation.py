from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('./titanic.csv')
df['male'] = df['Sex'] == 'male'
X = df[['Pclass', 'male', 'Age', 'Siblings/Spouses','Parents/Children']].values
y = df['Survived'].values

scores = []
kf = KFold(n_splits=5, shuffle=True)

for train_indices, test_indices in kf.split(X): 
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    model = LogisticRegression()
    model.fit(X_train, y_train)
    scores.append(model.score(X_test, y_test))

print(np.mean(scores))