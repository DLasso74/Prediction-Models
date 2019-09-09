# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 09:19:14 2019

House price prediction using a linear regression model.
Following the guidance of: https://towardsdatascience.com/create-a-model-to-predict-house-prices-using-python-d34fe8fad88f

@author: dlassog
"""

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

data = pd.read_csv('kc_house_data.csv')

model = LinearRegression()

labels = data['price']
conv_dates = [1 if values == 2010 else 0 for values in data.date]
data['date'] = conv_dates
train1 = data.drop(['id','price'],axis=1)
x_train, x_test, y_train, y_test = train_test_split(train1, labels, test_size = 0.40, random_state = 2)
model.fit(x_train,y_train)
score = model.score(x_test,y_test)
print("Model Accuracy = ",score)