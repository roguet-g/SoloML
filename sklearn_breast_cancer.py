from sklearn.datasets import load_breast_cancer
import pandas as pd

cancer_data = load_breast_cancer()

#print(cancer_data.keys())
#(cancer_data['DESCR'])

# print datapoints and attributes
#print(cancer_data['data'].shape)

#loading data into pandas
df = pd.DataFrame(cancer_data['data'], columns=cancer_data['feature_names'])
df['target'] = cancer_data['target']
print(df.head())