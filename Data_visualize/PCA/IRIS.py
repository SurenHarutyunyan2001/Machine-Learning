import matplotlib.pyplot as plt

# Неиспользуемый, но необходимый импорт для выполнения 3D проекций с matplotlib до версии 3.2
import mpl_toolkits.mplot3d  # noqa: F401

from sklearn import datasets
from sklearn.decomposition import PCA

# Импортируем данные для работы
iris = datasets.load_iris()
x = iris.data[:, :2]  # берём только первые два признака.
y = iris.target

x_min, x_max = x[:, 0].min() - 0.5, x[:, 0].max() + 0.5
y_min, y_max = x[:, 1].min() - 0.5, x[:, 1].max() + 0.5

plt.figure(2, figsize = (8, 6))
plt.clf()

# Строим график обучающих данных
plt.scatter(x[:, 0], x[:, 1], c = y, cmap = plt.cm.Set1, edgecolor = "k")
plt.xlabel("Длина чашелистика")
plt.ylabel("Ширина чашелистика")

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())

# Чтобы лучше понять взаимодействие признаков
# строим график для первых трёх компонент PCA
fig = plt.figure(1, figsize = (8, 6))
ax = fig.add_subplot(111, projection = "3d", elev = -150, azim = 110)

x_reduced = PCA(n_components = 3).fit_transform(iris.data)
ax.scatter(
    x_reduced[:, 0],
    x_reduced[:, 1],
    x_reduced[:, 2],
    c = y,
    cmap = plt.cm.Set1,
    edgecolor = "k",
    s = 40,
)

ax.set_title("Первые три направления PCA")
ax.set_xlabel("1-й собственный вектор")
ax.xaxis.set_ticklabels([])
ax.set_ylabel("2-й собственный вектор")
ax.yaxis.set_ticklabels([])
ax.set_zlabel("3-й собственный вектор")
ax.zaxis.set_ticklabels([])

plt.show()
