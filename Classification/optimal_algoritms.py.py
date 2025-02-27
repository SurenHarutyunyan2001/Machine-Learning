import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from matplotlib import pyplot as plt
from sklearn.svm import SVC 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV 
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA 
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier 
from sklearn.linear_model import LogisticRegression,Ridge

cancer = load_breast_cancer() 
x_train, x_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state = 0)

pipe = Pipeline([('preprocessing', StandardScaler()), ('classifier', SVC())]) 

param_grid = [     
    {'classifier': [SVC()], 'preprocessing': [StandardScaler(), None],      
     'classifier__gamma': [0.001, 0.01, 0.1, 1, 10, 100],      
     'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100]},     
     {'classifier': [RandomForestClassifier(n_estimators = 100)],      
      'preprocessing': [None], 
      'classifier__max_features': [1, 2, 3]}] 

grid = GridSearchCV(pipe, param_grid, cv = 5) 
grid.fit(x_train, y_train)

print("Наилучшие параметры:\n{}\n".format(grid.best_params_)) 
print("Наил значение правильности перекр проверки: {:.2f}".format(grid.best_score_)) 
print("Правильность на тестовом наборе: {:.2f}".format(grid.score(x_test, y_test))) 


