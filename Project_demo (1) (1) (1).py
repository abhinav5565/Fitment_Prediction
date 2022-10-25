#!/usr/bin/env python
# coding: utf-8

# In[3]:


#Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb#Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb


# In[4]:


data=pd.read_csv('Train.csv')


# In[5]:


data.head()


# In[6]:


data['BiasInfluentialFactor'].value_counts()


# In[7]:


data=data.drop(['EmpID','EmpName','JobProfileIDApplyingFor','GraduationYear'],axis=1)


# In[8]:


data.isnull().sum()


# In[9]:


data=data.replace(np.nan, 'No_Bias')


# In[10]:


data['BiasInfluentialFactor'].value_counts()


# In[11]:


data.isnull().sum()


# In[12]:


data = pd.get_dummies(data, columns = ['BiasInfluentialFactor'])


# In[13]:


data.head()


# In[14]:


data.isnull().sum()


# In[15]:


data['HighestDegree'].value_counts()


# In[16]:


data.head()


# In[17]:


data['DegreeBranch'].value_counts()


# In[18]:


data['GraduatingInstitute'].value_counts()


# In[19]:


data['CurrentDesignation'].value_counts()


# In[20]:


data['CurrentCompanyType'].value_counts()


# In[21]:


data['DepartmentInCompany'].value_counts()


# In[22]:


from sklearn import preprocessing
  
# label_encoder object knows how to understand word labels.
l1 = preprocessing.LabelEncoder()
  
# Encode labels in column 'species'.
l1.fit(data['LanguageOfCommunication'])
data['LanguageOfCommunication']= l1.transform(data['LanguageOfCommunication'])
  
data['LanguageOfCommunication'].unique()


# In[23]:


data.sample(10)


# In[24]:


data['LanguageOfCommunication'].value_counts()


# In[25]:


from sklearn import preprocessing
  
# label_encoder object knows how to understand word labels.
l2 = preprocessing.LabelEncoder()
  
# Encode labels in column 'species'.
l2.fit(data['Gender'])
data['Gender']= l2.transform(data['Gender'])
  
data['Gender'].unique()


# In[26]:


data['Gender'].value_counts()


# In[27]:


# Encode labels in column 'species'.
l3 = preprocessing.LabelEncoder()
l3.fit(data['HighestDegree'])
data['HighestDegree']= l3.transform(data['HighestDegree'])
  
data['HighestDegree'].unique()


# In[28]:


# Encode labels in column 'species'.
l4 = preprocessing.LabelEncoder()
l4.fit(data['DegreeBranch'])
data['DegreeBranch']= l4.transform(data['DegreeBranch'])
  
data['DegreeBranch'].unique()


# In[29]:


# Encode labels in column 'species'.
l5 = preprocessing.LabelEncoder()
l5.fit(data['GraduatingInstitute'])
data['GraduatingInstitute']= l5.transform(data['GraduatingInstitute'])
  
data['GraduatingInstitute'].unique()


# In[30]:


# Encode labels in column 'species'.
l6 = preprocessing.LabelEncoder()
l6.fit(data['MartialStatus'])
data['MartialStatus']= l6.transform(data['MartialStatus'])
  
data['MartialStatus'].unique()


# In[31]:


# Encode labels in column 'species'.
l7 = preprocessing.LabelEncoder()
l7.fit(data['CurrentDesignation'])
data['CurrentDesignation']= l7.transform(data['CurrentDesignation'])
  
data['CurrentDesignation'].unique()


# In[32]:


# Encode labels in column 'species'.
l8 = preprocessing.LabelEncoder()
l8.fit(data['CurrentCompanyType'])
data['CurrentCompanyType']= l8.transform(data['CurrentCompanyType'])
  
data['CurrentCompanyType'].unique()


# In[33]:


# Encode labels in column 'species'.
l9 = preprocessing.LabelEncoder()
l9.fit(data['DepartmentInCompany'])
data['DepartmentInCompany']= l9.transform(data['DepartmentInCompany'])
  
data['DepartmentInCompany'].unique()


# In[34]:


data.sample(10)


# In[35]:


sb.heatmap(data.corr());


# In[36]:


from sklearn.model_selection import train_test_split


# In[37]:


y=data['FitmentPercent']
x=data.drop('FitmentPercent',axis=1)


# In[38]:


y[:5]


# In[39]:


x[:5]


# In[40]:


# Split the data set into train and test
np.random.seed(0)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=3)


# In[41]:


from sklearn.linear_model import LinearRegression


# In[42]:


lm = LinearRegression()


# In[43]:


lm.fit(x_train,y_train)


# In[44]:


from sklearn.metrics import accuracy_score


# In[45]:


# The coefficients
print('Coefficients: \n', lm.coef_)


# In[46]:


predictions = lm.predict( x_test)


# In[47]:


plt.scatter(y_test,predictions)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')


# In[48]:


# calculate these metrics by hand!
from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# In[49]:


import statsmodels.api as sm
X_train_model = sm.add_constant(x_train)
lm_model2 = sm.OLS(y_train, X_train_model).fit()
lm_model2.summary()


# In[50]:


from sklearn.metrics import accuracy_score


# In[51]:


import numpy as np 
import pandas as pd
from sklearn.tree import DecisionTreeClassifier


# In[52]:


from sklearn.tree import DecisionTreeRegressor


# In[53]:


dt_model=DecisionTreeRegressor(random_state=100)


# In[54]:


dt_model.fit(x_train,y_train)


# In[55]:


dt_model.score(x_train,y_train)


# In[56]:


dt_model.score(x_test,y_test)


# In[57]:


dt_model.predict(x_test)[:10]


# In[58]:


y_test.head()


# In[59]:


# Fitting Random Forest Regression to the dataset
# import the regressor
from sklearn.ensemble import RandomForestRegressor
  
 # create regressor object
regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)
  
# fit the regressor with x and y data
regressor.fit(x_train, y_train)  


# In[60]:


regressor.score(x_train,y_train)


# In[61]:


regressor.score(x_test,y_test)


# In[62]:


y_pred=regressor.predict(x_test)


# In[63]:


for i in range(1,4):
  regressor = RandomForestRegressor(n_estimators = i, random_state = 0)
  regressor.fit(x_train, y_train) 
  print("Estimators value - {} Accuracy Score - {}".format(i,regressor.score(x_test,y_test)))


# In[64]:


regressor_1000 = RandomForestRegressor(n_estimators = 281, random_state = 100)
regressor_1000.fit(x_train, y_train) 
regressor_1000.score(x_test,y_test)


# In[65]:


import pickle
import joblib


# In[66]:


filename='fitment_pred.pkl'
joblib.dump(regressor_1000,open(filename,'wb'))


# In[67]:


#load model
loaded_model=joblib.load(open(filename,'rb'))


# In[68]:


loaded_model.score(x_test,y_test)


# In[69]:





# In[70]:


f1='output1.pkl'
joblib.dump(l1,open(f1,'wb'))
f2='output2.pkl'
joblib.dump(l2,open(f2,'wb'))
f3='output3.pkl'
joblib.dump(l3,open(f3,'wb'))
f4='output4.pkl'
joblib.dump(l4,open(f4,'wb'))
f5='output5.pkl'
joblib.dump(l5,open(f5,'wb'))
f6='output6.pkl'
joblib.dump(l6,open(f6,'wb'))
f7='output7.pkl'
joblib.dump(l7,open(f7,'wb'))
f8='output8.pkl'
joblib.dump(l8,open(f8,'wb'))
f9='output9.pkl'
joblib.dump(l9,open(f9,'wb'))


# In[71]:


#load model
df8=pd.read_csv("Train.csv")
loaded_model=joblib.load(open(filename,'rb'))
a=df8['LanguageOfCommunication'][0]
b=[]
b.append(a)
df5=pd.DataFrame({"LanguageOfCommunication":b})
df5['language']=l1.transform(df5['LanguageOfCommunication'])
print(df5['language'])
