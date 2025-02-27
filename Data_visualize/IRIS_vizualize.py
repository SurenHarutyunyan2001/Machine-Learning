import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
from pandas.plotting import scatter_matrix

iris = load_iris()

iris_df = pd.DataFrame(iris['data'], columns = iris['feature_names'])
iris_df['species'] = iris['target']

scatter_matrix(iris_df, alpha = 0.2, figsize = (10, 10))

plt.show()

print(iris['species'])
