#!/usr/bin/env python
# coding: utf-8

# In[2]:


import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


k=np.random.normal(25,10,200)
k.sort()


# In[5]:


k1=np.random.normal(15,5,200)
k1.sort()


# In[6]:


k2=np.random.normal(10,10,200)
k2.sort()


# In[7]:


g=np.random.normal(7.5,0.7,200)
g.sort()


# In[8]:


df=pd.DataFrame(k,columns=["English"])
df["Maths"]=k1
df["Science"]=k2
df["grade"]=g


# In[9]:


X=df.iloc[:,:-1]
Y=df.iloc[:,-1]


# In[10]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.3)


# In[11]:


x_train=x_train.values
y_train=y_train.values
x_test=x_test.values
y_test=y_test.values


# In[12]:


from sklearn.linear_model import LinearRegression


# In[13]:


model=LinearRegression()


# In[14]:


model.fit(x_train,y_train)


# In[15]:


y_pred=model.predict(x_test)


# In[16]:


from sklearn.metrics import r2_score
r2_score(y_test,y_pred)


# In[18]:


plt.scatter(y_test,y_pred,c="g")
plt.plot(model.predict(x_test),y_pred,c="b")


# In[ ]:





# In[ ]:




