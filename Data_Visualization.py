#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statistics as stat
import scipy.stats


# In[2]:


titanic = pd.read_csv('titanic.csv')


# In[3]:


titanic.shape


# In[4]:


titanic.head()


# In[5]:


titanic.tail()


# In[6]:


titanic.describe()


# In[7]:


titanic.columns


# In[8]:


titanic.nunique()


# In[9]:


titanic.isnull().sum()


# In[10]:


titanic.fillna(0,inplace = True)


# In[11]:


titanic.head()


# In[12]:


titanic.isnull().sum()


# In[13]:


titanic.tail()


# In[14]:


corelation = titanic.corr()


# In[15]:


sns.heatmap(corelation, xticklabels = corelation.columns, yticklabels = corelation.columns, annot=True)


# In[16]:


sns.pairplot(titanic)


# In[17]:


sns.relplot(x = 'Fare', y = 'Age', hue = 'Sex', data = titanic)


# In[18]:


sns.distplot(titanic['Age'],bins= 10)


# In[19]:


sns.histplot(titanic['Age'],bins= 5)


# In[20]:


mean = titanic['Age'].mean()
print('The mean is',mean)


# In[21]:


median = titanic['Age'].median()
print('The median is',median)


# In[22]:


mode = titanic['Age'].mode()
print('The mode is',mode)


# In[23]:


titanic1 = titanic.sort_values(by='Fare', ascending = False)


# In[24]:


titanic1.head()


# In[25]:


titanic2 = titanic1[titanic1.Survived == 1]


# In[26]:


titanic2.head()


# In[27]:


titanic2.shape


# In[28]:


plt.bar(x = titanic2['Age'], height = titanic2['Fare'])
plt.xlabel("Age")
plt.ylabel("Fare")
plt.show()


# In[29]:


##null hupothesis = avg age of people who survived titanic < 25
##alternate hypothesis = avg age of people who survived titanic => 25

##tcritical = tinverse(1-alpha , df)  df=n-1
##tstat = (avg - hyp mean)/(std dev/sqrt(n))
##central limit theorem

titanic4 = titanic[titanic.Survived ==1]
titanic4.count()


# In[30]:


Mean = titanic4['Age'].mean()
print("Mean:",Mean)
Std_dev = stat.stdev(titanic4['Age'])
print('Std_Dev',Std_dev)
Square_root = np.sqrt(titanic4['Age'].count())
print('Sqrt_Root',Square_root)

tstat = ((Mean-25))/(Std_dev/Square_root)
print('tstat', tstat)


# In[31]:


tcritical = scipy.stats.t.ppf(q = 0.95, df = 341)
print('tcritical',tcritical)


# In[32]:


if tstat > tcritical:
    print("Rejected null hypothesis. Alternate hypothesis holds true")
else:
    print("Failed to reject null hypothesis.")


# In[33]:


x = np.arange(-3, 3, 0.001)
plt.plot(x, scipy.stats.norm.pdf(x, 0, 1))
plt.axvline(x = tstat, color = 'r', label = 'axvline - full height')
plt.axvline(x = tcritical, color = 'y', label = 'axvline - full height')


# In[ ]:




