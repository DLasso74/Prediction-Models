# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 09:19:14 2019

House price prediction using a linear regression model.
Following the guidance of: https://towardsdatascience.com/create-a-model-to-predict-house-prices-using-python-d34fe8fad88f

@author: dlassog
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.read_csv('kc_house_data.csv')
labels = data['price']
data['year'] = pd.DatetimeIndex(data['date']).year
conv_dates = [1 if values == 2015 else 0 for values in data.year]
data['year'] = conv_dates
train1 = data.drop(['id','price','date'],axis=1)
x_train, x_test, y_train, y_test = train_test_split(train1, labels, test_size = 0.20, random_state = 2)

model = LinearRegression()
model.fit(x_train,y_train)
score = model.score(x_test,y_test)
print("Model Accuracy =",score,"%")