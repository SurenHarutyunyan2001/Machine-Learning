import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

x = np.array([6, 16, 26, 36, 46, 56]).reshape((-1, 1))
y = np.array([4, 23, 10, 12, 22, 35])
x_train,x_test,y_train,y_test = train_test_split(x, y, random_state = 0)

p = PolynomialFeatures(degree = 3).fit(x_train, y_train)
x_p = p.fit_transform(x)
pol = LinearRegression().fit(x_p, y)

plt.scatter(x,y)
plt.plot(x,pol.predict(p.fit_transform(x)))
plt.show()
