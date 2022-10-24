from sklearn.tree import export_graphviz
import graphviz
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv('./titanic.csv')
df['male'] = df['Sex'] == 'male'

feature_name = ['Pclass', 'male']

X = df[feature_name]
y = df['Survived']

dt = DecisionTreeClassifier()
dt.fit(X,y)

dot_file = export_graphviz(dt, feature_names=feature_name)
graph = graphviz.Source(dot_file)
graph.render(filename='tree', format='png', cleanup=True)
