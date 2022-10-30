from sklearn.datasets import load_digits

X, y = load_digits(n_class=2,return_X_y=True)

print(X.shape)
print(y.shape)
print(X[0].reshape(8,8))
print(y[0])