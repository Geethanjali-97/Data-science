#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import nltk
import re
import string
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer


# In[2]:


#Intialising
SW=stopwords.words('english')
tf_idf=TfidfVectorizer()
bow=CountVectorizer()


# In[3]:


#read data
pos_rev=pd.read_csv('pos.txt',sep='\n',header=None)

#changing the column name from 0--->Review
pos_rev.rename(columns={0:'review'},inplace=True)

#adding target
pos_rev['mood']=1


# In[4]:


pos_rev


# In[5]:


neg_rev=pd.read_csv('negative.txt',sep='\n',header=None)
neg_rev.rename(columns={0:'review'},inplace=True)
neg_rev['mood']=0


# In[6]:



# Text Preproccesing
"""
lower case
remove unwanted data
punctuation
stopwords
lemmatize - try yourself

"""


# In[7]:


# converter to lowercase
pos_rev['review'] = pos_rev['review'].apply(lambda x : x.lower())

# remove the numbers and hyphen
pos_rev['review'] = pos_rev['review'].apply(lambda x : re.sub(r"[0-9-]"," ",x)) # 0-9 removes numbers & - removes hyphen

# remove the '''
pos_rev['review'] = pos_rev['review'].apply(lambda x : re.sub(r"\W"," ",x)) #\W => not a word century's=> century s 

# remove the 's'
pos_rev['review'] = pos_rev['review'].apply(lambda x : re.sub(r"\b\w\b"," ",x)) 

#removing punctuation
pos_rev['review']=pos_rev['review'].apply(lambda x: " ".join([word for word in nltk.word_tokenize(x) if word not in string.punctuation]))

#stop words
pos_rev['review']=pos_rev['review'].apply(lambda x:" ".join([word for word in nltk.word_tokenize(x) if word not in SW]))

                                                                                                          
# In[8]:


neg_rev['review']=neg_rev['review'].apply(lambda x: x.lower())

neg_rev['review']=neg_rev['review'].apply(lambda x:re.sub(r"[0-9-]"," ",x))
neg_rev['review']=neg_rev['review'].apply(lambda x:re.sub(r"\W"," ",x))
neg_rev['review']=neg_rev['review'].apply(lambda x:re.sub(r"\b\w\b"," ",x))
neg_rev['review']=neg_rev['review'].apply(lambda x: " ".join([word for word in nltk.word_tokenize(x) if word not in string.punctuation]))    
neg_rev['review']=neg_rev['review'].apply(lambda x: " ".join([word for word in nltk.word_tokenize(x) if word not in SW]))


# In[9]:


#joining the data set
com_rev=pd.concat([pos_rev,neg_rev],axis=0,ignore_index=True)


# In[10]:


#split
X_train,X_test,y_train,y_test=train_test_split(com_rev['review'].values,com_rev['mood'],test_size=0.3,random_state=101)


# In[11]:


train_data=pd.DataFrame({'review':X_train,'mood':y_train})
test_data=pd.DataFrame({'review':X_test,'mood':y_test})


# In[12]:


train_vector=tf_idf.fit_transform(train_data['review'])
test_vector=tf_idf.transform(test_data['review'])


# In[13]:


#count vectorizer
bow_train=bow.fit_transform(train_data['review'])
bow_test=bow.transform(test_data['review'])


# In[14]:


train_vector


# In[15]:


#svm
from sklearn import svm
from sklearn.metrics import classification_report ,accuracy_score


# In[16]:


classifier1=svm.SVC()
classifier1.fit(train_vector,train_data['mood'])


# In[17]:


pred=classifier1.predict(test_vector)
accuracy_score(pred,test_data['mood'])


# In[18]:


# saving the model
import joblib
joblib.dump(classifier1, '75_netflix.pkl')
joblib.dump(tf_idf, 'tfidf.pkl')


# In[19]:


#loading the model
model=joblib.load('75_netflix.pkl')
tfidf=joblib.load('tfidf.pkl')


# In[31]:


sent=input('Enter the review')
vector=tfidf.transform([sent])
my_pred=model.predict(vector)
my_pred
 
if my_pred[0]==1:
    print('Positive')
else:
    print('Negative')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




