import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# Importing dataset
X, y = load_breast_cancer(return_X_y=True)

# Scaling data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Training
clf = KNeighborsClassifier()
clf.fit(X_train, y_train)
acc = clf.score(X_test, y_test)

print('Accuracy:', acc)
