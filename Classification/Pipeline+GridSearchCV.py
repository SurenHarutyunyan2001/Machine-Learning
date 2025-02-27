import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.svm import SVC 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV 


iris_dataset = load_iris() 
x_train, x_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state = 0) 

pipe = Pipeline([("scaler", MinMaxScaler()),("svm", SVC())])
param_grid = {'svm__C': [0.001, 0.01, 0.1, 1, 10, 100], 'svm__gamma': [0.001, 0.01, 0.1, 1, 10, 100]} 
grid = GridSearchCV(pipe, param_grid = param_grid, cv = 5) 
grid.fit(x_train, y_train) 
print("Наил значение правильности перекр проверки: {:.2f}".format(grid.best_score_)) 
print("Правильность на тестовом наборе: {:.2f}".format(grid.score(x_test, y_test))) 
print("Наилучшие параметры: {}".format(grid.best_params_)) 
