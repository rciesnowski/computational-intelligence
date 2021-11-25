#!/usr/bin/env python
# coding: utf-8

# In[2]:


pip install apyori


# In[16]:


import numpy as np
import pandas as pd
from apyori import apriori

book_data=pd.read_csv('titanic.csv', header=None)
book_data[[2, 4, 5, 1]]


# In[30]:


items=[]
for i in range(0,891):
    items.append([str(book_data.values[i,j]) for j in range(0,3)])
final_rule=apriori(items, min_support=0.05, min_confidence=0.7, min_lift=1.2, min_length=2)


# In[31]:


final_results=list(final_rule)
final_results


# In[ ]:




