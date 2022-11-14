#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Diwnload nedium or large model
get_ipython().system(' python -m spacy download en_core_web_md')


# In[12]:


import spacy
nlp=spacy.load("en_core_web_md")
text='The Republican president is being challenged by Democratic nominee Joe Biden vignesh '
doc=nlp(text)

for token in doc:
    print(token.text,'--->',token.has_vector)


# In[13]:


for token in doc:
    print(token.text,'--->',token.vector)


# In[14]:


text1='Men'
text2='Women'
doc1=nlp(text1)
doc2=nlp(text2)
score=doc1.similarity(doc2)
print(score)


# In[15]:


text3="lion cat tiger beer Men joe"
doc3=nlp(text3)
for token1 in doc3:
    for token2 in doc3:
        print(token1.text,token2.text,token1.similarity(token2))


# In[16]:


#Stanza
get_ipython().system(' pip install stanza')


# In[17]:


import stanza
stanza.download('en')


# In[19]:


#define stanza pipeline
nlp=stanza.Pipeline('en',use_gpu=True)


# In[23]:


doc=nlp('Barack obama was born in hawaii.Gandhi was born in India')
doc


# In[24]:


get_ipython().system(' pip install spacy.stanza')


# In[29]:


import spacy_stanza
nlp=spacy_stanza.load_pipeline('en')


# In[30]:


doc=nlp('Barack obama was born in hawaii.Gandhi was born in India')
doc


# In[32]:


for token in doc:
    print(token.text,token.pos_)
    


# In[36]:


for token in doc:
    print(token.text,'-->',token.lemma_)


# In[ ]:





# In[ ]:




