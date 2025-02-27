import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.svm import SVC 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, MinMaxScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline


iris_dataset = load_iris() 
x_train, x_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state = 0) 

pipe = Pipeline([("scaler", MinMaxScaler()), ("svm",SVC())])
pipe.fit(x_train, y_train)
print(pipe.score(x_test, y_test))
