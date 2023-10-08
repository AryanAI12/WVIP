#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
path = r'C:\Users\ARYAN SRIVASTAVA\OneDrive\Pictures\IMDB Dataset.csv'
df = pd.read_csv(path)
df.head()


# In[2]:


import numpy as np

from sklearn.feature_extraction.text import CountVectorizer

vect = CountVectorizer()

docs = np.array(['I am aryan who is making this project on sentiment analysis during my mid sem exams'])
bag = vect.fit_transform(docs)


# In[3]:


print(vect.vocabulary_)


# In[4]:


print(bag.toarray())


# In[11]:


import nltk
nltk.download('stopwords')


# In[6]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(
                        use_idf = True, 
                        norm = 'l2',
                        smooth_idf = True) 

y = df.sentiment.values
X = tfidf.fit_transform(df['review'].values.astype('U'))


# In[7]:


from sklearn.model_selection import train_test_split


# In[8]:


X_train,X_test,y_train,y_test = train_test_split(X,y, random_state=1, test_size = 0.5, shuffle=False)


# In[9]:


import pickle
from sklearn.linear_model import LogisticRegressionCV
clf = LogisticRegressionCV(cv=5,
                           scoring = 'accuracy',
                           random_state = 0,
                           n_jobs = -1,
                           verbose = 3,
                           max_iter=300).fit(X_train,y_train)

saved_model = open('saved_model.sav','wb')
pickle.dump(clf, saved_model)
saved_model.close()


# In[10]:


filename = 'saved_model.sav'
saved_clf = pickle.load(open(filename, 'rb')) 
saved_clf.score (X_test,y_test)


# In[ ]:




