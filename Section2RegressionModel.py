#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 17:22:48 2019

@author: fifi
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 12:43:55 2019

@author: fifi
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics


path = "/Users/fifi/Documents/City/AI/coursework"

filename_read = os.path.join(path, "AB_NYC_2019.csv")
df=pd.read_csv(filename_read, na_values=['NA','?','NaN'])
df = df.reindex(np.random.permutation(df.index))
df=df.interpolate(method='linear') #dealing with missing data -estimate the missing data from the available data points
"""
mask = np.random.rand(len(df)) < 0.8
trainDF = pd.DataFrame(df[mask])
validationDF = pd.DataFrame(df[~mask])
print(f"Training DF: {len(trainDF)}")
print(f"Validation DF: {len(validationDF)}")
"""
print(df.head(3))
print( list(df.columns))# show the features and label 
print( df.shape)
df.dtypes

df.isnull().sum()
df.fillna('0',inplace=True)

df.isnull().sum()


#Plots to understand the data better

#List of the columns containing numerical values 
numerical = [
   'latitude', 'longitude', 'price', 'minimum_nights', 'calculated_host_listings_count','availability_365', 'number_of_reviews', 'reviews_per_month'
]
#List of the columns containing categorical values 
categorical = [
  'neighbourhood_group', 'room_type']
#dropping columns that are not needed for the model
df.drop(columns=['name', 'host_id','host_name', 'neighbourhood','id','last_review',])
df = df[numerical + categorical]
df.shape


#Heatmap for the ralation between all the columns
relation = df.corr(method='kendall')
sns.heatmap(relation, annot=True)
df.columns

#Countplot for the count of airBnBs with the same price
sns.set(style='whitegrid', palette="deep", font_scale=1, rc={"figure.figsize": [8, 5]})
sns.distplot(
    df['price'], norm_hist=False, kde=False, bins=20, hist_kws={"alpha": 1}
).set(xlabel='Price', ylabel='Count');
df[numerical].hist(bins=15, figsize=(15, 6), layout=(2, 4));
#plot for room_type
sns.countplot(df['room_type']);
#plor for neighbourhood_group
sns.countplot(df['neighbourhood_group']);

#plot for the relation betweeb room_type and price
sns.scatterplot(x=df['room_type'], y=df['price']);
#plot for the relation betweeb neighbourhood_group and price
sns.scatterplot(x=df['neighbourhood_group'], y=df['price']);
#plot for the relation betweeb minimum_nights and price
sns.scatterplot(x=df['minimum_nights'], y=df['price']);
#plot for the relation betweeb availability_365 and price
sns.scatterplot(x=df['availability_365'], y=df['price']);

#More interesting plot for the relation between availability_365 and price showing both their individual plots and the relation
sns.jointplot(x=df['availability_365'], y=df['price']);

#Map for longtitude, latitude and neighbourhood_group
sns.scatterplot(df.longitude,df.latitude,hue=df.neighbourhood_group)
plt.ioff()

#Map for longtitude, latitude and room_type
sns.scatterplot(df.longitude,df.latitude,hue=df.room_type)
plt.ioff()


#Changing the categorical columns to dummy variables for better model
cacategoricalt_list = categorical
for value in cacategoricalt_list:
 add = pd.get_dummies(df[value], prefix=value)
 df1 = df.join(add)# join columns with old dataframe
 df = df1
#print bankdf.head(3)
#print bankdf.info()
 
#Converting headers into list
apart_vars = df.columns.values.tolist() 
#create a new list 
to_keep = [i for i in apart_vars if i not in cacategoricalt_list] 

#creating new DataFrame with the new data after converting Dummy variables 
apart_final = df[to_keep]

#New heatmap between the new columns (Dummy variables)
relation2 = apart_final.corr(method='kendall')
sns.heatmap(relation2, annot=True)
apart_final.columns



result = []
for x in apart_final.columns:
    if x != 'price':
        result.append(x)
   
X = df[result].values
y = df['price'].values


#Linear Regression Model


#split data into testing and training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

#fit the model
model = LinearRegression()  
model.fit(X_train, y_train)

print(model.coef_)

#Ppredictions for the linear Regression model
y_pred = model.predict(X_test)

#New dataframe containing actual and predicted values
df_compare = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df_head = df_compare.head(25)
print(df_head)

#Calculating mean and root of the Linear Regressor model
print('Mean of Linear Regressor:', np.mean(y_test))
print('Root Mean Squared Error of Linear Regressor:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

df_head.plot(kind='bar')
plt.show()

#Decision Tree Regressor
from sklearn.tree import DecisionTreeRegressor
#split data into testing and training
DX_train, DX_test, Dy_train, Dy_test = train_test_split(X, y, test_size=0.25, random_state=0)

#fit the model
DTree=DecisionTreeRegressor(min_samples_leaf=.0001)
DTree.fit(DX_train,Dy_train)

#Ppredictions for the linear Regression model
y_predict=DTree.predict(DX_test)

#New dataframe containing actual and predicted values
Ddf_compare = pd.DataFrame({'Actual': Dy_test, 'Predicted': y_predict})
Ddf_head = Ddf_compare.head(25)
print(Ddf_head)


print('Mean of Decision Tree Regressor:', np.mean(Dy_test))
print('Root Mean Squared Error of Decision Tree Regressor:', np.sqrt(metrics.mean_squared_error(Dy_test, y_predict)))

Ddf_head.plot(kind='bar')
plt.show()
