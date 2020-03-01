from sklearn.datasets import fetch_openml
mnist = fetch_openml("mnist_784", version=1)
#print(mnist.keys())

X, y = mnist["data"], mnist["target"]
print(len(X))
print(len(y))
import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np
y = y.astype(np.uint8)
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

from sklearn.neighbors import KNeighborsClassifier
y_train_large = (y_train>7)
y_train_odd = (y_train % 2 ==1)
y_multilabel = np.c_[y_train_large,y_train_odd]
knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_multilabel)
y_train_knn_pred = cross_val_predict(knn_clf, X_train, y_multilabel, cv=3)
print(f1_score(y_multilabel, y_train_knn_pred, average="macro"))

noise = np.random.randint(0,100,(len(X_train),784))
X_train_mod = X_train + noise
noise = np.random.randint(0,100,(len(X_test),784))
X_test_mod = X_test + noise
y_train_mod = X_train
y_test_mod = X_test

knn_clf.fit(X_train_mod, y_train_mod)
clean_digit = knn_clf.predict([X_test_mod[some_digit]])
# y_train_5 = (y_train==5)
# y_test_5 = (y_test==5)
#
# from sklearn.linear_model import SGDClassifier
# sgd_clf = SGDClassifier(random_state=42)
# # sgd_clf.fit(X_train, y_train_5)
# #print(sgd_clf.predict(X[0:5]))
#
# from sklearn.model_selection import cross_val_score
# #print(cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy"))
#
# # from sklearn.base import BaseEstimator
# # class Never5classifier(BaseEstimator):
# #     def fit(self, X, y=None):
# #         pass
# #     def predict(selfs, X):
# #         return np.zeros((len(X),1), dtype=bool)
# #
# # never5clf = Never5classifier()
# # print(cross_val_score(never5clf, X_train, y_train_5, cv=3, scoring="accuracy"))
#
# from sklearn.model_selection import cross_val_predict
# # y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
# # y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method = "decision_function")
# from sklearn.metrics import confusion_matrix
# # from sklearn.metrics import precision_score, recall_score, f1_score
# # from sklearn.metrics import precision_recall_curve
# # precisions, recalls, thresholds = precision_recall_curve(y_train_5,y_scores)
# #
# # def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
# #     plt.plot(thresholds, precisions[:-1], "b-")
# #     plt.plot(thresholds, recalls[:-1], "g--")
# # plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
# # plt.show()
#
#
# # print(confusion_matrix(y_train_5, y_train_pred))
# # print(precision_score(y_train_5, y_train_pred))
# # print(recall_score(y_train_5, y_train_pred))
# # print("f1_score:",f1_score(y_train_5, y_train_pred))
#
# some_digit = X[0]
# # y_scores = sgd_clf.decision_function([some_digit])
# # threshold = 8000
# # print(y_scores>threshold)
#
# # threshold_90 = thresholds[np.argmax(precisions >=0.90)]
# # print(threshold_90)
# # y_train_pred_90 = (y_scores>=threshold_90)
# # print(precision_score(y_train_5,y_train_pred_90))
# # print(recall_score(y_train_5,y_train_pred_90))
#
# from sklearn.metrics import roc_auc_score
# # print(roc_auc_score(y_train_5, y_scores))
#
# # from sklearn.ensemble import RandomForestClassifier
# # forest_clf = RandomForestClassifier(random_state=42)
# # y_train_pred_forest = cross_val_predict(forest_clf,X_train,y_train_5,cv=3)
# # y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5,cv=3, method="predict_proba")
# # y_scores_forest = y_probas_forest[:,1]
# #print(roc_auc_score(y_train_5,y_scores_forest))
# # print(precision_score(y_train_5,y_train_pred_forest))
# # print(recall_score(y_train_5,y_train_pred_forest))
#
# # from sklearn.svm import SVC
# # svm_clf = SVC()
# # svm_clf.fit(X_train, y_train)
# # svm_clf.predict([some_digit])
# # some_digit_scores = svm_clf.decision_function([some_digit])
# # print(some_digit_scores)
# # print(svm_clf.classes_)
# #
# # from sklearn.multiclass import OneVsRestClassifier
# # ovr_clf = OneVsRestClassifier(SVC())
# # ovr_clf.fit(X_train, y_train)
# # print(ovr_clf.decision_function([some_digit]))
# # print(cross_val_score(ovr_clf,X_train,y_train,cv=3,scoring="accuracy"))
#
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
# sgd_clf.fit(X_train, y_train)
# y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)
# conf_mx = confusion_matrix(y_train, y_train_pred)
# print(conf_mx)
# #print(cross_val_score(ovr_clf, X_train_scaled, y_train, cv=3, scoring="accuracy"))
#
# plt.matshow(conf_mx, cmap=plt.cm.gray)
# plt.show()