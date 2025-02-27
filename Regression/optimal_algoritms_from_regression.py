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
from sklearn.linear_model import LogisticRegression,Ridge

x = np.array([6, 16, 26, 36, 46, 56]).reshape((-1, 1))
y = np.array([4, 23, 10, 12, 22, 35])
x_train,x_test,y_train,y_test = train_test_split(x, y, random_state = 0)

pipe = make_pipeline(StandardScaler(),PolynomialFeatures(),Ridge()) 
param_grid = {'polynomialfeatures__degree': [1, 2, 3],'ridge__alpha': [0.001, 0.01, 0.1, 1, 10, 100]} 
grid = GridSearchCV(pipe, param_grid = param_grid, cv = 5, n_jobs = -1) 
grid.fit(x_train, y_train)

plt.matshow(grid.cv_results_['mean_test_score'].reshape(3, -1), vmin = 0, cmap="viridis") 
plt.xlabel("ridge__alpha") 
plt.ylabel("polynomialfeatures__degree") 
plt.xticks(range(len(param_grid['ridge__alpha'])), param_grid['ridge__alpha']) 
plt.yticks(range(len(param_grid['polynomialfeatures__degree'])), param_grid['polynomialfeatures__degree'])  
plt.colorbar()