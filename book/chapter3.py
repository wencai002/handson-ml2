from sklearn.datasets import fetch_openml
mnist = fetch_openml("mnist_784", version=1)
#print(mnist.keys())

X, y = mnist["data"], mnist["target"]
#print(X[0])
import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np
y = y.astype(np.uint8)
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

y_train_5 = (y_train==5)
y_test_5 = (y_test==5)

from sklearn.linear_model import SGDClassifier
sgd_clf = SGDClassifier(random_state=42)
#sgd_clf.fit(X_train, y_train_5)
#print(sgd_clf.predict(X[0:5]))

from sklearn.model_selection import cross_val_score
#print(cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy"))

# from sklearn.base import BaseEstimator
# class Never5classifier(BaseEstimator):
#     def fit(self, X, y=None):
#         pass
#     def predict(selfs, X):
#         return np.zeros((len(X),1), dtype=bool)
#
# never5clf = Never5classifier()
# print(cross_val_score(never5clf, X_train, y_train_5, cv=3, scoring="accuracy"))

from sklearn.model_selection import cross_val_predict
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
print(confusion_matrix(y_train_5, y_train_pred))
print(precision_score(y_train_5, y_train_pred))
print(recall_score(y_train_5, y_train_pred))
print("f1_score:",f1_score(y_train_5, y_train_pred))
