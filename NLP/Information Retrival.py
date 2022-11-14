#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import nltk
from sklearn.feature_extraction.text import CountVectorizer ,TfidfVectorizer
from sklearn.neighbors import KDTree


# In[2]:


person=pd.read_csv('famous_people.csv')
person


# In[3]:


tf_idf=TfidfVectorizer()
train_idf=tf_idf.fit_transform(person.Text).toarray()


# In[4]:


tf_idf.get_feature_names()


# In[5]:


person['tf_idf']=list(train_idf)
person


# In[6]:


kdtree=KDTree(train_idf)


# In[7]:


import joblib
joblib.dump(tf_idf,'tfidf_vector_model.pkl')
joblib.dump(kdtree,'kd_tree_model.pkl')


# In[8]:


distance, idx = kdtree.query(person.tf_idf[12].reshape(1,-1),k=3)
# (vector_distance , id)


# In[9]:


for i, value in list(enumerate(idx[0])):
    print(f"Name: {person['Name'][value]}")
    print(f"URI: {person['URI'][value]}")


# In[10]:


vector=joblib.load('tfidf_vector_model.pkl')
kdtree=joblib.load('kd_tree_model.pkl')


# In[11]:


text=input('search')
transform=vector.transform([text]).toarray()


# In[13]:


distance,idx=kdtree.query(transform,k=3)
for i ,value in list(enumerate(idx[0])):
    print(f"Name: {person['Name'][value]}")
    print(f"URI: {person['URI'][value]}")
    


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




