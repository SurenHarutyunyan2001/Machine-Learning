import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier  

iris_dataset = load_iris() 
x_train, x_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state = 0) 

knn = KNeighborsClassifier(n_neighbors = 1)    
knn.fit(x_train, y_train) 

x_new = np.array([[5, 2.9, 1, 0.2]]) 
print("форма массива x_new: {}".format(x_new.shape)) 
prediction = knn.predict(x_new) 
print("Прогноз: {}".format(prediction)) 
print("Спрогнозированная метка: {}".format(iris_dataset['target_names'][prediction])) 

y_pred = knn.predict(x_test)
print("Правильность на тестовом наборе: {:.2f}".format(np.mean(y_pred == y_test))) 
print("Правильность на тестовом наборе: {:.2f}".format(knn.score(x_test, y_test))) 





