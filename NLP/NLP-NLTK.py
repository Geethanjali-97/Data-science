#!/usr/bin/env python
# coding: utf-8

# In[20]:


get_ipython().system('pip install nltk')


# In[21]:


import nltk
nltk.download('punkt')


# In[22]:


data='According to consensus in modern genetics, anatomically modern humans first arrived on the Indian subcontinent from Africa between 73,000 and 55,000 years ago.[1] However, the earliest known human remains in South Asia date to 30,000 years ago. Settled life, which involves the transition from foraging to farming and pastoralism, began in South Asia around 7000 BCE. At the site of Mehrgarh presence can be documented of the domestication of wheat and barley, rapidly followed by that of goats, sheep, and cattle.[2] By 4500 BCE, settled life had spread more widely,[2] and began to gradually evolve into the Indus Valley civilisation, an early civilisation of the Old World, which was contemporaneous with Ancient Egypt and Mesopotamia. This civilisation flourished between 2500 BCE and 1900 BCE in what today is Pakistan and north-western India, and was noted for its urban planning, baked brick houses, elaborate drainage, and water supply.[3]'


# In[23]:


nltk.sent_tokenize(data)


# In[24]:


word_tokens=nltk.word_tokenize(data)
word_tokens


# In[25]:


#POS_tag

nltk.download('averaged_perceptron_tagger')


# In[26]:


nltk.pos_tag(word_tokens)


# In[27]:


nltk.download('stopwords')
from nltk.corpus import stopwords
SW=stopwords.words('english')
print(SW)


# In[28]:


import string
punct=string.punctuation
punct


# 
# 
# 
# 

# In[29]:


from nltk.stem import PorterStemmer,LancasterStemmer,SnowballStemmer
lancaster=LancasterStemmer()
porter=PorterStemmer()
snow=SnowballStemmer('english')

print('LancasterStemmer')
print(lancaster.stem('finally'))
print(lancaster.stem('final'))
print(lancaster.stem('computer'))
print(lancaster.stem('computerization'))

print('PorterStemmer')
print(porter.stem('finally'))
print(porter.stem('final'))
print(porter.stem('computer'))
print(porter.stem('computerization'))

print('SnowballStemmer')
print(snow.stem('finally'))
print(snow.stem('final'))
print(snow.stem('computer'))
print(snow.stem('computerization'))
  


# In[30]:


sent='i was going to the office on my bike when i saw a car passing by hit the tree'
token=list(nltk.word_tokenize(sent))
for stemmer in (snow,porter,lancaster):
    print(stemmer)
    stemm=[stemmer.stem(t) for t in token]
    print(" ".join(stemm))


# In[31]:


nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
lemma=WordNetLemmatizer()
print(lemma.lemmatize('running',pos='v'))
print(lemma.lemmatize('runs',pos='v'))
print(lemma.lemmatize('ran',pos='v'))


# In[32]:


nltk.download('maxent_ne_chunker')
nltk.download('words')


# In[33]:


sent='It was in the 17th century that the Europeans came to India. This coincided with the disintegration of the Mughal Empire, paving the way for regional states. In the contest for supremacy, the English emerged'


# In[34]:


word=nltk.word_tokenize(sent)
pos_tag=nltk.pos_tag(word)


# In[35]:


named_entity=nltk.ne_chunk(pos_tag)


# In[36]:


print(named_entity)


# In[37]:


from nltk import FreqDist


# In[38]:


words=nltk.word_tokenize(sent)
freq=FreqDist(words)
print(freq)


# In[41]:


freq.plot()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




