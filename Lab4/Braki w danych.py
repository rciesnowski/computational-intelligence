#!/usr/bin/env python
# coding: utf-8

# In[205]:


import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn import neighbors, svm, datasets
from sklearn.impute import SimpleImputer
from sklearn.model_selection import learning_curve


# In[206]:


iris = datasets.load_iris()
X = iris.data[:, :2]
y = iris.target
X,y = shuffle(X, y, random_state=0)

X_naned = X.copy()
zmienione = []
for i in range (np.random.randint(20,50)):
    w = np.random.randint(150)
    kol = np.random.randint(2)
    X_naned[w][kol] = np.nan
    zmienione.append(w)
    i += 1
    
y_clean = np.delete(y, zmienione)
X_clean = np.delete(X, zmienione, axis=0)

imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp.fit(X_naned)
X_imputed = imp.transform(X_naned)


# In[207]:


def zrob(X,y):
    regresja_log = LogisticRegression(C=1e5)
    regresja_log.fit(X, y)

    jadro_svm = svm.SVC(kernel='linear', C=1e5)
    jadro_svm.fit(X, y)

    k_naj = neighbors.KNeighborsClassifier(8, weights="uniform")
    k_naj.fit(X, y)

    
    metody = [(regresja_log, "r", "regresja logistyczna", 0), (jadro_svm, "b", "jadro SVM", 1), (k_naj, "g", "k-najblizsi", 2)]

    plt.figure(figsize= (20,10))
    plt.grid()
    plt.ylabel("skutecznosc")
    plt.xlabel("liczba probek")
    wynik = 0
    for metoda, kolor, nazwa, i in metody:
        train_sizes, train_scores, test_scores = learning_curve(metoda, X, y, n_jobs=-1, cv=None, train_sizes=np.linspace(0.1, 1, ), verbose=0)
        test_scores_mean = np.mean(test_scores, axis=1)
        wynik += test_scores_mean[-1]
        plt.plot(train_sizes, test_scores_mean, color=kolor, label=nazwa)
    print("Skuteczność: " + "{:.2f}".format(wynik*100/3) + "%")
    plt.legend()
    plt.show()


# In[208]:


# oryginalne dane
zrob(X,y)


# In[209]:


# dane w których usunieto zepsute wiersze
zrob(X_clean, y_clean)


# In[210]:


# dane w których naprawiono zepsute wiersze
zrob(X_imputed, y)

