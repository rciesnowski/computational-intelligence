#!/usr/bin/env python
# coding: utf-8

# In[1]:


a = 123
b = 321
a*b


# In[7]:


#suma wektorów
w1 = [3,8,9,10,12]
w2 = [8,7,7,5,6]
[a + b for a, b in zip(w1, w2)]


# In[10]:


#iloczyn wektorów
[a*b for a,b in zip(w1,w2)]


# In[11]:


#iloczyn skalarny
import numpy as np
np.dot(w1, w2)


# In[14]:


#odleglosci euklidesowe
np.sqrt(np.sum([(a - b)**2 for a,b in zip(w1,w2)]))


# In[53]:


#dwie dowolne macierze 3x3
m1 = np.random.randint(10, size = (3,3))
m2 = np.random.randint(10, size=(3,3))
print(m1)
print(m2)


# In[51]:


#pomnożenie po wspolrzednych
np.multiply(m1,m2)


# In[52]:


#pomnozenie macierzowo
np.dot(m1,m2)


# In[46]:


#dowolny wektor
import random
w50 = [random.randrange(1, 100, 1) for a in range(50)]


# In[33]:


#srednia
sum(w50)/50


# In[34]:


#najmniejszy
min(w50)


# In[35]:


max(w50)


# In[36]:


#odchylenie standardowe
np.std(w50)


# In[54]:


#Dokonaj normalizacji wektora z podpunktu (e) (ściskamy wszystkie liczby do przedziału [0,1]) za pomocą poniższego wzoru (xi to liczba w starym wektorze na pozycji i, a zi to liczba w nowym wektorze na pozycji i)
[(a - min(w50))/(max(w50)-min(w50)) for a in w50]


# In[58]:


import pandas as pd
data = pd.read_csv('https://inf.ug.edu.pl/~gmadejsk/io-pliki-2021/miasta.csv')
data


# In[67]:


data.append(pd.DataFrame([['2010', '460', '555', '405']], columns = data.columns), ignore_index = True)


# In[82]:


import matplotlib.pyplot as plt
data.plot(marker = 'o',x='Rok', y="Gdansk", color='red')
plt.ylabel('Ludnosc w tysiacach')
plt.show()


# In[80]:


ax = plt.gca()
data.plot(x='Rok', y="Poznan", ax=ax)
data.plot(x='Rok', y="Gdansk", ax=ax)
data.plot(x='Rok', y = 'Szczecin', ax=ax)
plt.ylabel('Ludnosc w tysiacach')
plt.show()


# In[ ]:




