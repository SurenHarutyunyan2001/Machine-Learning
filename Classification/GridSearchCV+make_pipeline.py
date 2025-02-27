import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.svm import SVC 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV 
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA 
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression 

cancer = load_breast_cancer() 
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state = 0)

pipe = make_pipeline(StandardScaler(), LogisticRegression()) 
param_grid = {'logisticregression__C': [0.01, 0.1, 1, 10, 100]} 
grid = GridSearchCV(pipe, param_grid, cv = 5) 
grid.fit(X_train, y_train) 
print("Лучшая модель:\n{}".format(grid.best_estimator_)) 
print("Этап логистической регрессии:\n{}".format(grid.best_estimator_.named_steps["logisticregression"])) 
print("Коэффициенты логистической регрессии:\n{}".format(       grid.best_estimator_.named_steps["logisticregression"].coef_)) 