from matplotlib.colors import ListedColormap
from sklearn.datasets import make_moons
from sklearn.cluster import KMeans
import numpy as np
from sklearn.model_selection import KFold

# Генерируем случайные данные
X, y = make_moons(n_samples=1000, noise=0.1)

# Создаем модель KMeans с двумя кластерами
kmeans = KMeans(n_clusters=2)

# Обучаем модель на данных
kmeans.fit(X)

# Получим метки кластеров для каждого объекта
y_pred = kmeans.labels_

# Оценим качество кластеризации
accuracy = np.mean(y == y_pred)
print(f"Accuracy: {accuracy:.2f}")

import matplotlib.pyplot as plt

# разделение объектов на два класса на основе их меток
class_0 = X[y == 0]
class_1 = X[y == 1]

# визуализация данных
plt.scatter(class_0[:, 0], class_0[:, 1], color='blue', label='Class 0')
plt.scatter(class_1[:, 0], class_1[:, 1], color='red', label='Class 1')
plt.legend(loc='best')
plt.title('Data Visualization')
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()

import pymc3 as pm
import theano.tensor as tt

with pm.Model() as model:
    # Определяем гиперпараметры сети
    n_hidden = 10
    n_features = 2

    # Определяем распределения параметров весов
    # и смещений для каждого слоя
    weights_in = pm.Normal('weights_in', mu=0, sd=1,
                           shape=(n_features, n_hidden))
    bias_in = pm.Uniform('bias_in', -1, 1, shape=n_hidden)

    weights_out = pm.Normal('weights_out', mu=0, sd=1,
                            shape=n_hidden)
    bias_out = pm.Uniform('bias_out', -1, 1)

    # Определяем скрытый слой с функцией активации tanh
    hidden = tt.tanh(tt.dot(X, weights_in) + bias_in)

    # Определяем выходной слой
    output = tt.nnet.sigmoid(tt.dot(hidden, weights_out) + bias_out)

    # Определяем распределение Бернулли для меток классов
    y_obs = pm.Bernoulli('y_obs', p=output, observed=y)

    # Запускаем алгоритм ADVI для обучения модели
    approx = pm.fit(n=1000000, method='advi', n_init=10)

     # Обучаем модель на текущем фолде
    trace = approx.sample(draws=100000)

# Получаем распределение вероятностей, полученное из байесовской нейронной сети
with model:
    ppc = pm.sample_posterior_predictive(trace, samples=1000000, progressbar=True)['y_obs']
y_nn = ppc.mean(axis=0)

# Выводим гистограммы распределений
import matplotlib.pyplot as plt
fig, axs = plt.subplots(1, 2, figsize=(10, 4))
axs[0].hist(y, bins=2, alpha=0.5)
axs[0].set(title='True distribution', xlabel='Class label', ylabel='Frequency')
axs[1].hist(y_nn, bins=2, alpha=0.5)
axs[1].set(title='Neural network distribution', xlabel='Class label', ylabel='Frequency')
plt.show()

weights_in = trace["weights_in"].mean(axis=0)
bias_in = trace["bias_in"].mean(axis=0)
weights_out = trace["weights_out"].mean(axis=0)
bias_out = trace["bias_out"].mean()

# Создаем новую модель с параметрами из выборки
with pm.Model() as test_model:
    hidden = tt.tanh(tt.dot(X, weights_in) + bias_in)
    output = tt.nnet.sigmoid(tt.dot(hidden, weights_out) + bias_out)
    y_pred = pm.Bernoulli('y_pred', p=output, shape=y.shape, testval=np.random.rand(*y.shape))

# Получаем прогнозы на тестовых данных
with test_model:
    ppc = pm.sample_posterior_predictive(trace, var_names=["y_pred"])

# Визуализируем результаты
fig, axs = plt.subplots(1, 2, figsize=(10, 4))
axs[0].scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm")
axs[0].set_title("True labels")
axs[1].scatter(X[:, 0], X[:, 1], c=ppc["y_pred"].mean(axis=0), cmap="coolwarm")
axs[1].set_title("Predicted labels")
plt.show()

class_0 = X[y_nn < 0.49999999999]
class_1 = X[y_nn >= 0.5]

plt.scatter(class_0[:, 0], class_0[:, 1], color='blue', label='Class 0')
plt.scatter(class_1[:, 0], class_1[:, 1], color='red', label='Class 1')

plt.legend(loc='best')
plt.title('Prediction Results')
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()


# Извлекаем выборку из апостериорного распределения
weights_in_trace = trace['weights_in']
bias_in_trace = trace['bias_in']
weights_out_trace = trace['weights_out']
bias_out_trace = trace['bias_out']

# Определяем функцию для прогнозирования меток классов
def predict(X):
    # Считаем среднее значение параметров сети
    weights_in_mean = weights_in_trace.mean(axis=0)
    bias_in_mean = bias_in_trace.mean(axis=0)
    weights_out_mean = weights_out_trace.mean(axis=0)
    bias_out_mean = bias_out_trace.mean(axis=0)

    # Определяем скрытый слой с функцией активации tanh
    hidden = np.tanh(np.dot(X, weights_in_mean) + bias_in_mean)

    # Определяем выходной слой
    output = 1 / (1 + np.exp(-(np.dot(hidden, weights_out_mean) + bias_out_mean)))

    # Возвращаем метки классов
    return output > 0.5

# Оцениваем качество классификации на обучающей выборке
y_pred = predict(X)
accuracy = (y_pred == y).mean()
print(f'Accuracy on training set: {accuracy:.2f}')

# Строим график разделяющей поверхности
h = .02
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#0000FF'])
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
plt.scatter(X[:, 0], X[:, 1], c=y_nn, cmap=cmap_bold)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Classification results')
plt.show()


