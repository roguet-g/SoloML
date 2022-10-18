from sklearn.datasets import load_breast_cancer
import pandas as pd
from sklearn.linear_model import LogisticRegression

cancer_data = load_breast_cancer()

#print(cancer_data.keys())
#(cancer_data['DESCR'])

# print datapoints and features
#print(cancer_data['data'].shape)

#loading data into pandas
df = pd.DataFrame(cancer_data['data'], columns=cancer_data['feature_names'])
df['target'] = cancer_data['target']
#print(df.head())

#preparing the regression data
X = df[cancer_data.feature_names].values
y = df['target'].values

model = LogisticRegression(solver='liblinear')
model.fit(X,y)
model.predict([X[0]])
print(model.score(X,y))