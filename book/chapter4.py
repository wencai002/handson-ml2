import numpy as np
# X = 2*np.random.rand(100,1)
# y = 4 + 3*X + np.random.randn(100,1)
# X_ = np.c_[np.ones((100,1)),X]
# theta_best = np.linalg.inv(X_.T.dot(X_)).dot(X_.T).dot(y)
#
#
# X_new = np.array([[2],[0]])
# X_new_ = np.c_[np.ones((2,1)),X_new]
# y_pred = X_new_.dot(theta_best)
#
import matplotlib.pyplot as plt
#
# # plt.plot(X_new,y_pred,"r-")
# # plt.plot(X,y, "g.")
# # plt.show()
#
from sklearn.linear_model import LinearRegression
# lin_reg = LinearRegression()
# lin_reg.fit(X,y)
# ### least square
# theta_best_lstsq, residuals, rank, s = np.linalg.lstsq(X_,y,rcond=1e-6)
# ###pseudoinverse
# print(np.linalg.pinv(X_).dot(y))

# n_epochs = 50
# t0, t1 = 5, 50
# m = 100
# def learning_schedule(t):
#     return t0/(t+t1)
#
# theta = np.random.rand(2,1)
# for epoch in range(n_epochs):
#     for i in range(m):
#        random_index = np.random.randint(m)
#        xi = X_[random_index:random_index+1]
#        yi = y[random_index:random_index+1]
#        gradients = 2*xi.T.dot(xi.dot(theta)-yi)
#        eta = learning_schedule(epoch*m+i)
#        theta = theta - eta*gradients
#
from sklearn.linear_model import SGDRegressor
# sgd_reg = SGDRegressor(max_iter=1000, tol=1e-3, penalty=None, eta0=0.1)
# sgd_reg.fit(X,y.ravel()) # .ravel makes column to array
# print(sgd_reg.intercept_, sgd_reg.coef_)

m = 100
X = 6*np.random.rand(m,1)-3
y = 0.5*X**2 + X + 2 + np.random.rand(m,1)

from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)
lin_reg = SGDRegressor(max_iter=1000, tol=1e-5, penalty=None)
lin_reg.fit(X_poly,y.ravel())
print(lin_reg.intercept_,lin_reg.coef_)

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
def plot_learning_curve(model, X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    train_errors, val_errors = [], []
    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train[:m], y_train_predict))
        val_errors.append(mean_squared_error(y_val, y_val_predict))
    plt.plot(np.sqrt(train_errors), "r-", linewidth=2, label="train")
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="val")
    plt.ylim(0,3)
    plt.show()

# lin_reg = LinearRegression()
# plot_learning_curve(lin_reg,X,y)

from sklearn.pipeline import Pipeline
# polynomial_regression = Pipeline([
#     ("poly_features", PolynomialFeatures(degree=10,include_bias=False)),
#      ("lin_reg", LinearRegression())
# ])
# plot_learning_curve(polynomial_regression, X,y)

from sklearn.linear_model import Ridge
ridge_reg = Ridge(alpha=1, solver="cholesky")
ridge_reg.fit(X,y)
print(ridge_reg.predict([[1.5]]))

sgd_reg = SGDRegressor(penalty="l2")
sgd_reg.fit(X,y.ravel())
print(sgd_reg.predict([[1.5]]))

from sklearn.linear_model import ElasticNet
ela_net = ElasticNet(alpha=0.1, l1_ratio=0.5)
ela_net.fit(X, y)
print(ela_net.predict([[1.5]]))

poly_scaler = Pipeline([
    ("poly_features", PolynomialFeatures(degree=100, include_bias=False)),
    ("std_sca", StandardScaler())
])
X_train_poly_scaled = poly_scaler.fit_transform(X_train)
X_val_poly_scaled = poly_scaler.transform(X_val)

from sklearn.base import clone
sgd_reg = SGDRegressor(max_iter=1, tol=-np.infty, warm_start=True, penalty=None, learning_rate="constant", eta0=0.0005)
# warm start is used to train continously
minimum_val_error = float("inf")
best_epoch = None
best_model = None
# for epoch in range(1000):
#     sgd_reg.fit(X_train_poly_scaled, y_train.ravel())
#     y_val_predict = sgd_reg.predict(X_val_poly_scaled)
#     val_error = mean_squared_error(y_val, y_val_predict)
#     if val_error < minimum_val_error:
#         minimum_val_error = val_error
#         best_epoch = epoch
#         best_model = clone(sgd_reg)
#         print(best_epoch)

from sklearn import datasets
iris = datasets.load_iris()
X = iris["data"][:,3:]
y = (iris["target"]==2).astype(np.int)

from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(X,y)

X_new = np.linspace(0,3,1000).reshape(-1,1) # creae 1000 points between 0 and 3
y_prob = log_reg.predict_proba(X_new)
plt.plot(X_new,y_prob[:,1],"g-")
plt.plot(X_new,y_prob[:,0],"b--")
plt.show()