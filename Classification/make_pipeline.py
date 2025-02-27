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

cancer = load_breast_cancer() 
x_train, x_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state = 0) 

pipe_short = make_pipeline(MinMaxScaler(), SVC(C = 100))
print("Этапы конвейера:\n{}".format(pipe_short.steps)) 

pipe = make_pipeline(StandardScaler(), PCA(n_components = 2), StandardScaler()) 
print("Этапы конвейера:\n{}".format(pipe.steps)) 

pipe.fit(cancer.data)
components = pipe.named_steps["pca"].components_ 
print("форма components: {}".format(components.shape)) 
