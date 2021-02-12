"""
Classifying Iris species
"""

import numpy as np
import pandas as pd

# Import data
iris = pd.read_csv("Iris.csv")

iris = iris.drop('Id', axis = 1)
print(iris.head())

# Basic data exploration
print('Informacion del dataset:')
print(iris.info())

print('Estadisticas del dataset:')
print(iris.describe())

print('Distribucion de datos por especie:')
print(iris.groupby('Species').size())

# Visualize the variables 
import matplotlib.pyplot as plt

# Sepal - Length vs Width
fig = iris[iris.Species == 'Iris-setosa'].plot(kind = 'scatter',
            x = 'SepalLengthCm', y = 'SepalWidthCm', color = 'blue', label = 'Setosa')
iris[iris.Species == 'Iris-versicolor'].plot(kind = 'scatter',
      x = 'SepalLengthCm', y = 'SepalWidthCm', color = 'green', label = 'Versicolor', ax = fig)
iris[iris.Species == 'Iris-virginica'].plot(kind = 'scatter',
      x = 'SepalLengthCm', y = 'SepalWidthCm', color = 'red', label = 'Virginica', ax = fig)

fig.set_xlabel('Sepalo - Longitud')
fig.set_ylabel('Sepalo - Ancho')
fig.set_title('Sepalo - Longitud vs Ancho')
plt.show()

# Petal - Length vs Width
fig = iris[iris.Species == 'Iris-setosa'].plot(kind = 'scatter',
            x = 'PetalLengthCm', y = 'PetalWidthCm', color = 'blue', label = 'Setosa')
iris[iris.Species == 'Iris-versicolor'].plot(kind = 'scatter',
      x = 'PetalLengthCm', y = 'PetalWidthCm', color = 'green', label = 'Versicolor', ax = fig)
iris[iris.Species == 'Iris-virginica'].plot(kind = 'scatter',
      x = 'PetalLengthCm', y = 'PetalWidthCm', color = 'red', label = 'Virginica', ax = fig)

fig.set_xlabel('Petalo - Longitud')
fig.set_ylabel('Petalo - Ancho')
fig.set_title('Petalo - Longitud vs Ancho')
plt.show()

# ML algorithms - default parameters

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# Model with all data
X = np.array(iris.drop(['Species'],1))
y = np.array(iris['Species'])

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
print('Son {} datos para entrenamiento y {} datos para prueba'.format(X_train.shape[0], X_test.shape[0]))

# ML Algorithms

algoritmo = LogisticRegression()
algoritmo.fit(X_train, y_train)
Y_pred = algoritmo.predict(X_test)
print('Precision Regresion Logistica: {}'.format(algoritmo.score(X_train, y_train)))

algoritmo = SVC()
algoritmo.fit(X_train, y_train)
Y_pred = algoritmo.predict(X_test)
print('Precision Maquinas de Soporte Vectorial: {}'.format(algoritmo.score(X_train, y_train)))

algoritmo = KNeighborsClassifier(n_neighbors = 5)
algoritmo.fit(X_train, y_train)
Y_pred = algoritmo.predict(X_test)
print('Precision de K Vecinos mas cercanos: {}'.format(algoritmo.score(X_train, y_train)))

algoritmo = DecisionTreeClassifier()
algoritmo.fit(X_train, y_train)
Y_pred = algoritmo.predict(X_test)
print('Precision del Arbol de descicion: {}'.format(algoritmo.score(X_train, y_train)))
