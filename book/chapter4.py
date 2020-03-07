import numpy as np
X = 2*np.random.rand(100,1)
y = 4 + 3*X + np.random.randn(100,1)
X_ = np.c_[np.ones((100,1)),X]
# theta_best = np.linalg.inv(X_.T.dot(X_)).dot(X_.T).dot(y)
#
#
# X_new = np.array([[2],[0]])
# X_new_ = np.c_[np.ones((2,1)),X_new]
# y_pred = X_new_.dot(theta_best)
#
# import matplotlib.pyplot as plt
#
# # plt.plot(X_new,y_pred,"r-")
# # plt.plot(X,y, "g.")
# # plt.show()
#
# from sklearn.linear_model import LinearRegression
# lin_reg = LinearRegression()
# lin_reg.fit(X,y)
# ### least square
# theta_best_lstsq, residuals, rank, s = np.linalg.lstsq(X_,y,rcond=1e-6)
# ###pseudoinverse
# print(np.linalg.pinv(X_).dot(y))

n_epochs = 50
t0, t1 = 5, 50
m = 100
def learning_schedule(t):
    return t0/(t+t1)

theta = np.random.rand(2,1)
for epoch in range(n_epochs):
    for i in range(m):
       random_index = np.random.randint(m)
       xi = X_[random_index:random_index+1]
       yi = y[random_index:random_index+1]
       gradients = 2*xi.T.dot(xi.dot(theta)-yi)
       eta = learning_schedule(epoch*m+i)
       theta = theta - eta*gradients

from sklearn.linear_model import SGDRegressor
sgd_reg = SGDRegressor(max_iter=1000, tol=1e-3, penalty=None, eta0=0.1)
sgd_reg.fit(X,y.ravel()) # .ravel makes column to array
print(sgd_reg.intercept_, sgd_reg.coef_)