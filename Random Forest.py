#!/usr/bin/env python
# coding: utf-8

# In[2]:


import time
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import roc_curve,auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold,cross_val_score
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from urllib.request import urlopen 

plt.style.use('ggplot')
pd.set_option('display.max_columns', 500) 
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


Breast_cancer=pd.read_csv('data.csv')
Breast_cancer.head(5)


# In[4]:


Breast_cancer.set_index('id')


# In[5]:


encoding=preprocessing.LabelEncoder()
Breast_cancer['diagnosis']=encoding.fit_transform(Breast_cancer['diagnosis'])
Breast_cancer['diagnosis']


# # Missing Value checks and Data Cleaning

# In[6]:


Breast_cancer.isnull().sum()


# In[7]:


del Breast_cancer['Unnamed: 32']


# In[8]:


Breast_cancer


# In[9]:


Breast_cancer.describe()


# # Split the Dataset for Training & Testing

# In[10]:


x=Breast_cancer.iloc[:,Breast_cancer.columns!='diagnosis']

y=Breast_cancer.iloc[:,Breast_cancer.columns=='diagnosis']


# In[11]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=42)


# In[12]:


# Cleaning test sets to avoid future warning messages
y_train = y_train.values.ravel() 

y_test = y_test.values.ravel() 


# ## Random Forest Classifier

# In[13]:


# Set the random state for reproducibility
rf_fit=RandomForestClassifier(random_state=42)


# In[14]:


## Hyper-Parameter Optimization using GridSearchCV


# In[15]:


np.random.seed(42)
start=time.time()
param_dist={'max_depth':[2,3,4],
            'bootstrap': [True, False],
            'max_features': ['auto','sqrt','log2', None],
            'criterion': ['gini','entropy']
           }
cv_rf=GridSearchCV(rf_fit,cv=10,param_grid=param_dist,n_jobs=3)
cv_rf.fit(x_train,y_train)
print('Best Parameters using grid search: \n', cv_rf.best_params_)
end = time.time()
print('Time taken in grid search: {0: .2f}'.format(end - start))


# In[16]:


# Set best parameters given by grid search 
rf_fit.set_params(criterion = 'entropy',
                  max_features = 'log2', 
                  max_depth = 4)


# In[17]:


rf_fit.set_params(warm_start=True, 
                  oob_score=True)

min_estimators = 20
max_estimators = 1000

error_rate = {}

for i in range(min_estimators, max_estimators + 1):
    rf_fit.set_params(n_estimators=i)
    rf_fit.fit(x_train,y_train)

    oob_error = 1 - rf_fit.oob_score_
    error_rate[i] = oob_error


# In[18]:


error_rate


# In[19]:


# Convert dictionary to a pandas series for easy plotting 
oob_series = pd.Series(error_rate)


# In[20]:


oob_series.plot(kind='line',color = 'red')
plt.xlabel('n_estimators')
plt.ylabel('OOB Error Rate')
plt.title('OOB Error Rate Across various Forest sizes \n(From 20 to 1000 trees)')


# In[21]:


print('OOB Error rate for 450 trees is: {0:.5f}'.format(oob_series[450]))


# In[22]:


# Refine the tree via OOB Output
rf_fit.set_params(n_estimators=450,
                  bootstrap = True,
                  warm_start=False, 
                  oob_score=False)


# # Train the RandomForest

# In[23]:


rf_fit.fit(x_train,y_train)


# ## Predictions
# 

# In[27]:



predictions_rf=rf_fit.predict(x_test)



# In[32]:


conf_matrix=confusion_matrix(y_test,predictions_rf)
sns.heatmap(conf_matrix,annot=True,fmt='d',cbar=False)

plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.title('Actual vs. Predicted Confusion Matrix')
plt.show()


# ## Accuracy

# In[35]:



Accuracy_rf=rf_fit.score(x_test,y_test)
Accuracy_rf


# In[42]:


from sklearn import metrics
Accuracy=metrics.accuracy_score(y_test,predictions_rf)
Accuracy


# ## Area Under Curve (AUC)

# In[43]:


y_pred_proba =rf_fit.predict_proba(x_test)[::,1]
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




