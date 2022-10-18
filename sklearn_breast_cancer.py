from sklearn.datasets import load_breast_cancer

cancer_data = load_breast_cancer()

print(cancer_data.keys())
print(cancer_data['DESCR'])