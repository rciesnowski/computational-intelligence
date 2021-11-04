#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn import datasets
from scipy.sparse import csr_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import VarianceThreshold

faces = datasets.fetch_olivetti_faces()
data = StandardScaler().fit_transform(faces.data)
target = faces.target

pca = PCA(n_components = 0.9, whiten = True)
lda = LinearDiscriminantAnalysis(n_components = 30)
thresh = VarianceThreshold(threshold=1)

data_pca = pca.fit_transform(data)
data_lda = lda.fit(data, target).transform(data)
data_thresh = thresh.fit_transform(data)

print("poczatkowa liczba cech:", data.shape[1])
print("po redukcji PCA:", data_pca.shape[1])
print("po redukcji LDA:", data_lda.shape[1])
print("po redukcji Treshold:", data_thresh.shape[1])


# In[4]:


import matplotlib.pyplot as plt
import numpy as np
from sklearn import neighbors, datasets, svm
from sklearn.model_selection import learning_curve
from sklearn.linear_model import LogisticRegression

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
        train_sizes, train_scores, test_scores = learning_curve(metoda, X, y, n_jobs=-1, cv=None, train_sizes=np.linspace(0.59, 1, ), verbose=0)
        test_scores_mean = np.mean(test_scores, axis=1)
        wynik += test_scores_mean[-1]
        plt.plot(train_sizes, test_scores_mean, color=kolor, label=nazwa)
    print("Skuteczność: " + "{:.2f}".format(wynik*100/3) + "%")
    plt.legend()
    plt.show()


# In[49]:


zrob(data, target)


# In[50]:


zrob(data_lda, target)


# In[5]:


zrob(data_pca, target)


# In[52]:


zrob(data_thresh, target)


# In[ ]:




