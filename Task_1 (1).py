#!/usr/bin/env python
# coding: utf-8

# In[3]:


#Import the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
#import LinearRegression Model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


# In[5]:


#Import Dataset 
url = 'http://bit.ly/w-data'
data = pd.read_csv(url)
data.head(10)


# In[6]:


# checking if a dataset has any missing value :
data.isnull().sum()


# In[7]:


#moro info about dataset :
data.describe()


# In[14]:


data.plot(kind='scatter',x='Hours',y='Scores')
plt.title('Hours vs Scores')
plt.show()


# In[21]:


#Splitting the dataset into Train and Test set
x = data.iloc[:,:-1]
y = data.iloc[:,-1]
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)
print(X_train.shape)
print(X_test.shape)


# In[23]:


#Training Dataset with Linear Regression Model
Regressor = LinearRegression()
Regressor.fit(X_train,Y_train)


# In[27]:


#predicting the results
Y_pred = Regressor.predict(X_test)


# In[37]:


#Measure Testing data accuracy
print('Testing Accuracy = ',Regressor.score(X_test,Y_test)*100,'%')


# In[51]:


pd.DataFrame({'True Value':Y_test,'Predict Value':Y_pred})


# In[55]:


#Predict score if a student studies for 8.5 hrs/day
Predict_score = Regressor.predict([[8.5]])
print(int(Predict_score))


# In[52]:


#Calculate Mean Absolute Error
#Calculate Mean Squared Error
#Calculate Root Mean Absolute Error

print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, Y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(Y_test, Y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, Y_pred)))


# In[50]:


#Visualising the Test set results
plt.scatter(X_test,Y_test,color = 'black')
plt.plot(X_test,Y_pred, color = 'red')
plt.title('Hours vs Score')
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.show()


# In[ ]:




