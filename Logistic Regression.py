#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn import metrics

get_ipython().run_line_magic('matplotlib', 'inline')
import warnings;
warnings.filterwarnings('ignore');


# In[3]:


df=pd.read_csv('Social_Network_Ads.csv')


# In[4]:


df.head()


# In[5]:


df


# In[6]:


X=df.iloc[:,2:4]
X


# In[7]:


Y=df.iloc[:,4].values
Y


# In[8]:


sns.heatmap(df.corr(),annot = True,cmap='RdYlGn')


# In[9]:


X_train,x_test,Y_train,y_test=train_test_split(X,Y,train_size=0.7,random_state=32)


# In[10]:


#Feature Scaling


# In[11]:


standard_Scaler=StandardScaler()
X_train = standard_Scaler.fit_transform(X_train)
x_test = standard_Scaler.transform(x_test)


# In[12]:


standard_Scaler=StandardScaler()
X_train=standard_Scaler.fit_transform(X_train)
x_test = standard_Scaler.transform(x_test)


# In[13]:


X_train


# In[14]:


#Instantiating and fitting the model to training Dataset


# In[15]:


log_reg=LogisticRegression(random_state=1)
log_reg.fit(X_train,Y_train)


# In[16]:


y_pred=log_reg.predict(x_test)
y_pred


# In[17]:


y_test


# In[18]:


#Visualizing the Training Set Result


# In[19]:


X_set,y_set = X_train,Y_train
X1,X2 = np.meshgrid(np.arange(start=X_set[:,0].min() - 1,stop=X_set[:,0].max()+1,step=0.01),
                    np.arange(start=X_set[:,1].min() - 1,stop=X_set[:,1].max()+1,step=0.01))
plt.contourf(X1,X2,log_reg.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),
                        alpha=0.75,cmap=ListedColormap(('black','red')))

plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set ==j,0],X_set[y_set == j,1],
               c=ListedColormap(['blue','green'])(i),label=j)

plt.title('Logistic Regression (Train set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()


# In[20]:


#Visualizing the Testing Set results


# In[21]:


X_set,y_set = x_test,y_test
X1,X2 = np.meshgrid(np.arange(start=X_set[:,0].min() - 1,stop=X_set[:,0].max()+1,step=0.01),
                    np.arange(start=X_set[:,1].min() - 1,stop=X_set[:,1].max()+1,step=0.01))
plt.contourf(X1,X2,log_reg.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),
                        alpha=0.75,cmap=ListedColormap(('black','blue')))

plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set ==j,0],X_set[y_set == j,1],
               c=ListedColormap(['yellow','red'])(i),label=j)

plt.title('Logistic egression (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()


# In[22]:


from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(y_test,y_pred)
conf_matrix


# In[23]:


accuracy=(65+35)/len(y_test)
accuracy


# In[24]:


##### Mis Classification Rate
mis_cla_rate  = (11+6)/len(y_test)
mis_cla_rate


# In[25]:


#Accuracy, Precision, Recall etc


# In[26]:


print("Accuracy: ", metrics.accuracy_score(y_test,y_pred))


# In[28]:


print("Precision: ", metrics.precision_score(y_test,y_pred))


# In[29]:


print("Recall: ", metrics.recall_score(y_test,y_pred))


# In[30]:


#zROC and AUC


# In[31]:


y_pred_proba = log_reg.predict_proba(x_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.figure(figsize=(10,8))
plt.plot(fpr,tpr,label="auc="+str(auc))
plt.legend(loc=4)
plt.title("Receiver Operating Characteristic Curve (ROC)")
plt.xlabel("FPR ---->")
plt.ylabel("TPR ---->")
plt.show()


# In[ ]:





# In[ ]:




