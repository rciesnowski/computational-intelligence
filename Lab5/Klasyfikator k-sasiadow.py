#!/usr/bin/env python
# coding: utf-8

# In[5]:


import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import neighbors, svm, datasets


# In[6]:


iris = datasets.load_iris()
X = iris.data[:, :2]
y = iris.target


# In[7]:


regresja_log = LogisticRegression(C=1e5)
regresja_log.fit(X, y)

jadro_svm = svm.SVC(kernel='linear', C=1e5)
jadro_svm.fit(X, y)

k_naj = neighbors.KNeighborsClassifier(5, weights="uniform")
k_naj.fit(X, y)


# In[8]:


from sklearn.model_selection import learning_curve
metody = [(regresja_log, "r", "regresja logistyczna", 0), (jadro_svm, "b", "jadro SVM", 1), (k_naj, "g", "k-najblizsi", 2)]

fig, axs = plt.subplots(1, 3, figsize=(20,10))
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
h = .02
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
for metoda, kolor, nazwa, i in metody:
    uruchom = metoda.predict(np.c_[xx.ravel(), yy.ravel()])
    uruchom = uruchom.reshape(xx.shape)
    axs[i].pcolormesh(xx, yy, uruchom, cmap=plt.cm.cool, shading='auto')
    axs[i].scatter(X[:, 0], X[:, 1], edgecolors='k', cmap=plt.cm.cool)
    axs[i].set(xlabel='dlugosc', ylabel='szerokosc')
    axs[i].title.set_text(nazwa)
plt.show()

plt.figure(figsize= (20,10))
plt.grid()
plt.ylabel("skutecznosc")
plt.xlabel("liczba probek")
for metoda, kolor, nazwa, i in metody:
    train_sizes, train_scores, test_scores = learning_curve(metoda, X, y, n_jobs=-1, cv=None, train_sizes=np.linspace(.1, 1.0, 5), verbose=0)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.plot(train_sizes, test_scores_mean, 'o-', color=kolor, label=nazwa)
plt.legend()
plt.show()

