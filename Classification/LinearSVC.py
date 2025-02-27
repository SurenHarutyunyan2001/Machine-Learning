import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC


iris_dataset = load_iris() 
x_train, x_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state = 0) 

model = LinearSVC(C = 10)
model.fit(x_train, y_train) 
x_new = np.array([[5, 2.9, 1, 0.2]]) 
print("форма массива X_new: {}".format(x_new.shape)) 
prediction = model.predict(x_new) 
print("Прогноз: {}".format(prediction)) 
print("Спрогнозированная метка: {}".format(iris_dataset['target_names'][prediction])) 

print("train score",model.score(x_train,y_train))
print("test score",model.score(x_test,y_test))
