import matplotlib.pyplot as plt
import seaborn as sns   
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans


x, y_true = make_blobs(n_samples = 300, centers = 4,cluster_std = 0.60, random_state = 0)
plt.scatter(x[:, 0], x[:, 1], s = 50)
plt.show()


kmeans = KMeans(n_clusters = 4)
kmeans.fit(x)
y_kmeans = kmeans.predict(x)

plt.scatter(x[:, 0], x[:, 1], c = y_kmeans, s = 50, cmap = 'viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c = 'black', s = 200, alpha = 0.5)
plt.show()