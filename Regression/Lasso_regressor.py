import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split


x = np.array([6, 16, 26, 36, 46, 56]).reshape((-1, 1))
y = np.array([4, 23, 10, 12, 22, 35])
x_train,x_test,y_train,y_test = train_test_split(x, y, random_state = 0)


model = Lasso(alpha = 0.1, max_iter = 100000).fit(x_train, y_train)
print("(w)/coefficents", model.coef_)
print("(b)/offset", model.intercept_)
print(model.score(x_train, y_train))
print(model.score(x_test, y_test))
print(model.predict(x_test))

y_predict = model.predict(x)
plt.scatter(x,y)
plt.plot(x,y_predict)
plt.show()


x_new = np.array([[40]])
y_new = model.predict(x_new)
print("y_new= ",y_new)
