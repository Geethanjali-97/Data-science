#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


# In[10]:


ad=pd.read_csv('Advertising.csv')


# In[11]:


ad.head()


# In[13]:


ad.isna().sum()


# In[14]:


ad.describe()


# In[ ]:





# In[15]:


ad.info()


# In[16]:


ad.shape


# In[21]:


#vizualization
sns.pairplot(ad,x_vars=['TV','Radio','Newspaper'],y_vars=['Sales'],size=5,kind='scatter')


# In[27]:


sns.heatmap(ad.corr(),cmap="PuBuGn_r",annot=True)


# In[41]:


ad=ad.drop('Unnamed: 0' ,axis=1)


# In[64]:


X=ad.iloc[:,:3].values
Y=ad.iloc[:,-1:].values


# In[65]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,train_size=0.7,test_size=0.3,random_state=101)


# In[78]:


lr=LinearRegression()
lr_m=lr.fit(X_train,Y_train)
Coef=lr_m.coef_
print(Coef)
Inter=lr_m.intercept_
print(Inter)


# In[95]:


y_train_pred=lr_m.predict(X_train)
res=Y_train-y_train_pred


# In[96]:


sns.distplot(res,bins=15)
plt.title('Error Terms')
plt.xlabel('Residual Errors')
plt.ylabel('freq')


# In[97]:


y_test_pred=lr_m.predict(X_test)


# In[98]:


from sklearn.metrics import r2_score

r_sq = r2_score(Y_test, y_test_pred)
r_sq


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




