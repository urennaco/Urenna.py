#!/usr/bin/env python
# coding: utf-8

# In[21]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import sklearn.linear_model
import ipywidgets
from ipywidgets import interact
from pandas.plotting import scatter_matrix


# # Exploring the data

# In[22]:


diamond=pd.read_csv(r'C:\Users\urenn\OneDrive\Desktop\wholesale_diamonds.csv')
print("Number of rows=%i"%len(diamond))
print("Number of columns=%i"%len(diamond.columns))
diamond.head()


# In[23]:


#rename columns:Cost,Length,Width and height and summarize statistics
diamond.rename(columns ={'cost (dollars)':'price', 'length (mm)':'x', 'width (mm)':'y', 'height (mm)':'z'}, inplace = True)
diamond.describe()


# In[24]:


#Check for Nan Value
count_nan=diamond.isnull().sum()
print(count_nan[count_nan>0])


# In[25]:


#To drop nan value in carat and negative value in price
diamond=diamond.dropna(axis=0, how='any')
diamond.drop(diamond[diamond['price'] <= 0].index, inplace = True)
diamond.shape


# In[26]:


# drop columns with only 1 unique value
diamond = diamond[[c for c
             in list(diamond)
             if len(diamond[c].unique()) > 1]]
diamond.shape


# In[27]:


diamond.describe()


# In[28]:


diamond.corr()


# In[29]:


print(diamond.info())


# In[30]:


diamond.hist(bins=50,figsize=(20,15))


# In[31]:


corr= diamond.corr()
f, ax = plt.subplots(figsize=(15,12))
sns.heatmap(corr,cmap="Accent",annot=True)


# # Building Model: Price Prediction

# In[32]:


# to create variable that specify a target value
y=diamond.price
print(y)


# In[33]:


#To create variable that holds the predictive features
feature_names=['carat', 'cut','color','clarity', 'depth', 'table','x','y','z']
feature_names = diamond.drop(["price","index","year"], axis = 1)
X = feature_names
X.describe()


# In[34]:


from sklearn.preprocessing import LabelEncoder
# apply the encoder on cut, color and clarity to read
encoder = LabelEncoder()
# select_if(is.object, col)
to_encode = X.select_dtypes(exclude = 'float').columns.values
X[list(to_encode)] = X[to_encode].apply(encoder.fit_transform)
X.describe()


# In[36]:


#To specify and fit model
from sklearn.tree import DecisionTreeRegressor
diamond_model= DecisionTreeRegressor(random_state=1)
diamond_model.fit(X, y)


# In[37]:


#To make predictions
y_pred=diamond_model.predict(X)
y_pred=np.rint(y_pred)
print("The predictions are")
print(y_pred[:50])


# In[40]:


#Calculating the mean absolute error for model validation
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
y_pred= diamond_model.predict(X)
mae=mean_absolute_error(y,y_pred)
mape=mean_absolute_percentage_error(y,y_pred)*100
print("The MAE of the model is::", mae)
print("The MAPE of the Predict_model is::", mape)


# In[93]:



from sklearn.linear_model import LinearRegression
linreg=LinearRegression()
linreg.fit(X,y)
linreg.intercept_


# In[94]:


linreg.coef_


# In[102]:


linreg.predict(X[:10])


# In[64]:


#To split data into training and validation 
from sklearn.model_selection import train_test_split
train_X,val_X,train_y,val_y=train_test_split(X,y,random_state=1)
diamond_model=DecisionTreeRegressor()
diamond_model.fit(train_X,train_y)
yval_predict=diamond_model.predict(val_X)
ytrain_predict=diamond_model.predict(train_X)
mae= mean_absolute_error(val_y,yval_predict)
mape_val=mean_absolute_percentage_error(val_y,yval_predict)*100
mape_train=mean_absolute_percentage_error(train_y,ytrain_predict)*100
print("The MSE of the model is::", mae)
print("The MAPE of the test model is::", mape_val)
print("The MAPE of the train model is::", mape_train)
print(len(train_X),len(val_X), len(train_y), len(val_y) )


# # Predicting with Diamond_for_sale_2022

# In[47]:


sdiamond=pd.read_csv(r'C:\Users\urenn\OneDrive\Desktop\diamonds_for_sale_2022.csv')
print("Number of rows=%i"%len(sdiamond))
print("Number of columns=%i"%len(sdiamond.columns))
sdiamond.head()


# In[48]:


#rename columns:Cost,Length,Width and height and summarize statistics
sdiamond.rename(columns ={'cost (dollars)':'price', 'length (mm)':'x', 'width (mm)':'y', 'height (mm)':'z'}, inplace = True)
sdiamond.describe()


# In[49]:


#Check for Nan Value
count_nan=sdiamond.isnull().sum()
print(count_nan[count_nan>0])


# In[50]:


sdiamond.shape


# In[51]:


sdiamond=sdiamond.dropna(axis=0, how='any')
# drop columns with only 1 unique value
sdiamond = sdiamond[[c for c
             in list(sdiamond)
             if len(sdiamond[c].unique()) > 1]]
sdiamond.shape


# In[52]:


sdiamond.head()


# In[53]:


feature1_names=['carat', 'cut','color','clarity', 'depth', 'table','x','y','z']
XX=sdiamond[feature1_names]
XX.describe()


# In[54]:


XX.dtypes


# In[55]:


#To create variable that holds the predictive features
from sklearn.preprocessing import LabelEncoder
# apply the encoder on cut, color and clarity to read
encoder = LabelEncoder()
# select_if(is.object, col)
to_encode = XX.select_dtypes(exclude = 'float').columns.values
XX[list(to_encode)] = XX[to_encode].apply(encoder.fit_transform)
XX.describe()


# In[56]:


y_pred=diamond_model.predict(XX)
y_pred=np.rint(y_pred)
print("The predictions are")
print(y_pred[:50])


# # Random Forest

# In[57]:


from sklearn.ensemble import RandomForestRegressor


# In[112]:


forest_model=RandomForestRegressor()
forest_model.fit(train_X,train_y)
yvall_predict=forest_model.predict(val_X)
ytrainn_predict=forest_model.predict(train_X)
print("The MAE of the model is",mean_absolute_error(val_y,yvall_predict))
mape_vall=mean_absolute_percentage_error(val_y,yvall_predict)*100
mape_trainn=mean_absolute_percentage_error(train_y,ytrainn_predict)*100
print("The MAPE of the test model is::", mape_vall)
print("The MAPE of the train model is::", mape_trainn)
print(len(train_X),len(val_X), len(train_y), len(val_y) )


# In[74]:





# In[ ]:




