import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Importing dataset
X, y = load_iris(return_X_y=True)

# Scaling data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Training
clf = LogisticRegression(solver='lbfgs',multi_class='multinomial')
clf.fit(X_train, y_train)
acc = clf.score(X_test, y_test)

print('Accuracy:', acc)