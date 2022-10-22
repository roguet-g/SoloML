import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, precision_recall_fscore_support

sensitivity_score = recall_score

def specificity_score(y_true, y_pred):
    p, r,f,s = precision_recall_fscore_support(y_true, y_pred)
    return r[0]

df = pd.read_csv('./titanic.csv')
df['male'] = df['Sex'] == 'male'
X = df[['Pclass', 'male', 'Age', 'Siblings/Spouses','Parents/Children']]
y = df['Survived'].values

X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=5)

model = LogisticRegression()
model.fit(X_train, y_train)
#y_pred = model.predict(X_test)

y_pred = model.predict_proba(X_test)[:,1] > 0.75
print('Sensitivity : ', sensitivity_score(y_test, y_pred))
print('specificity ; ', specificity_score(y_test, y_pred))
#print(model.predict_proba(X_test))

print('precision : ', precision_score(y_test, y_pred))
print('recall : ', recall_score(y_test, y_pred))

