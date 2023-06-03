#!/usr/bin/env python
# coding: utf-8

# ## Data Cleaning

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
sb.set()


# In[2]:


data = pd.read_csv('Raw Data.csv')
data


# In[3]:


data.columns


# In[4]:


data.columns = ['feed_rate', 'depth_of_cut', 'ultrasonic_vibration', 'cutting_fluid', 'cutting_force', 'surface_roughness']
data.columns


# In[5]:


data.info()


# In[6]:


data['feed_rate'] = data['feed_rate'].astype('float')
data['depth_of_cut'] = data['depth_of_cut'].astype('float')


# In[7]:


data.dtypes


# In[8]:


data['ultrasonic_vibration'] = data['ultrasonic_vibration'].apply(lambda x:1 if 'On' in x else 0)
data['cutting_fluid'] = data['cutting_fluid'].apply(lambda x:1 if 'On' in x else 0)


# In[9]:


data.head()


# In[10]:


data['ultrasonic_vibration'].value_counts()


# In[11]:


data['cutting_fluid'].value_counts()


# ## Exploratoty Data Analysis

# In[12]:


plt.figure(figsize = [20, 5])
plt.subplot(1,2,1)
sb.histplot(data['cutting_force'], bins=15, kde=True)

plt.subplot(1,2,2)
sb.histplot(np.log(data['cutting_force']), bins=15, kde=True);


# In[13]:


plt.figure(figsize = [20, 5])
plt.subplot(1,2,1)
sb.histplot(data['surface_roughness'], bins=15, kde=True)

plt.subplot(1,2,2)
sb.histplot(np.log(data['surface_roughness']), bins=15, kde=True);


# The output variables will be on logarithmic scale to have normal distribution 

# In[14]:


numeric_vars = ['feed_rate', 'depth_of_cut', 'ultrasonic_vibration', 'cutting_fluid', 'cutting_force', 'surface_roughness']

plt.figure(figsize = [6, 2])
corr_matrix = data[numeric_vars].corr()
mask = np.zeros_like(corr_matrix)
mask[np.triu_indices_from(mask)] = True
sb.heatmap(corr_matrix, mask=mask, annot = True, fmt = '.2f', cmap = 'vlag_r', center = 0)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.show()


# Only cutting fluid is correlated with cutting force and surface roughness, so we had to sub divide the data according to the boolean variables unltrasonic vibration and cutting fluid.

# In[15]:


plt.figure(figsize = [15, 4])
plt.subplot(1,2,1)
sb.histplot(data['feed_rate'], bins=10)

plt.subplot(1,2,2)
sb.histplot(data['depth_of_cut'], bins=10);


# In[16]:


plt.figure(figsize = [9, 8])

plt.subplot(2,2,1)
sb.barplot(x=data['cutting_fluid'], y=data['cutting_force'])

plt.subplot(2,2,2)
sb.barplot(x=data['cutting_fluid'], y=data['surface_roughness'])

plt.subplot(2,2,3)
sb.barplot(x=data['ultrasonic_vibration'], y=data['cutting_force'])

plt.subplot(2,2,4)
sb.barplot(x=data['ultrasonic_vibration'], y=data['surface_roughness']);


# In[17]:


sb.lmplot(data=data, x='feed_rate', y='cutting_force', hue='cutting_fluid', col='ultrasonic_vibration', height=5, aspect=1.3);


# In[18]:


sb.lmplot(data=data, x='feed_rate', y='surface_roughness', hue='cutting_fluid', col='ultrasonic_vibration', height=5, aspect=1.3);


# In[19]:


sb.lmplot(data=data, x='depth_of_cut', y='cutting_force', hue='cutting_fluid', col='ultrasonic_vibration', height=5, aspect=1.3);


# In[20]:


sb.lmplot(data=data, x='depth_of_cut', y='surface_roughness', hue='cutting_fluid', col='ultrasonic_vibration', height=5, aspect=1.3);


# In[21]:


plt.figure(figsize = [20, 6])
plt.subplot(1,2,1)
sb.scatterplot(data=data, x='feed_rate', y='cutting_force', hue='ultrasonic_vibration')

plt.subplot(1,2,2)
sb.scatterplot(data=data, x='feed_rate', y='surface_roughness', hue='ultrasonic_vibration');


# After subdividing the data according to the boolean variables, there is a correlation between numerical variables (feed rate and depth of cut) and output variables when subdividing the cutting fluid boolean variable. In contrast, there is no clear correlation when subdividing the untrasonic vibration boolean variable. so, untrasonic vibration has no great effect on output variables.

# In[22]:


data_cf0 = data[data['cutting_fluid'] == 0]
numeric_vars = ['feed_rate', 'depth_of_cut', 'ultrasonic_vibration', 'cutting_force', 'surface_roughness']

plt.figure(figsize = [6, 2])
corr_matrix = data_cf0[numeric_vars].corr()
mask = np.zeros_like(corr_matrix)
mask[np.triu_indices_from(mask)] = True
sb.heatmap(corr_matrix, mask=mask, annot = True, fmt = '.2f', cmap = 'vlag_r', center = 0)
plt.xticks(fontsize=9)
plt.yticks(fontsize=9)
plt.show()


# In[23]:


data_cf1 = data[data['cutting_fluid'] == 1]
numeric_vars = ['feed_rate', 'depth_of_cut', 'ultrasonic_vibration', 'cutting_force', 'surface_roughness']

plt.figure(figsize = [6, 2])
corr_matrix = data_cf1[numeric_vars].corr()
mask = np.zeros_like(corr_matrix)
mask[np.triu_indices_from(mask)] = True
sb.heatmap(corr_matrix, mask=mask, annot = True, fmt = '.2f', cmap = 'vlag_r', center = 0)
plt.xticks(fontsize=9)
plt.yticks(fontsize=9)
plt.show()


# ## Splitting the data into train and test sets

# In[24]:


columns_to_drop = ['cutting_force', 'surface_roughness']
targets = np.log(data[['cutting_force', 'surface_roughness']])
inputs = data.drop(columns_to_drop, axis=1)
x_train, x_test, y_train, y_test = train_test_split(inputs, targets, test_size=0.2, random_state=1)


# In[25]:


x_test


# In[26]:


x_train.to_csv('x_train.csv')
x_test.to_csv('x_test.csv')
y_train.to_csv('y_train.csv')
y_test.to_csv('y_test.csv')

