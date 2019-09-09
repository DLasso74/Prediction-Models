# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 09:19:14 2019

House price prediction using a gradient boosting regressor model.
Following the guidance of: https://towardsdatascience.com/create-a-model-to-predict-house-prices-using-python-d34fe8fad88f

@author: dlassog
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import ensemble

data = pd.read_csv('kc_house_data.csv')
labels = data['price']
data['yearsold'] = pd.DatetimeIndex(data['date']).year
conv_dates = [1 if values == 2015 else 0 for values in data.yearsold]
data['yearsold'] = conv_dates
data['new'] = conv_dates
for i in range(21613):
    built = int(data.loc[i,['yr_built']])
    remodeled = int(data.loc[i,['yr_renovated']])
    if built > remodeled :
        check = built
    else:
        check = remodeled
    if check >= 2011:
        data.iloc[i,22] = 1
    else:
        data.iloc[i,22] = 0
train1 = data.drop(['id','price','date'],axis=1)
x_train, x_test, y_train, y_test = train_test_split(train1, labels, test_size = 0.20, random_state = 2)

model = ensemble.GradientBoostingRegressor(n_estimators = 375, max_depth = 5, min_samples_split = 3, learning_rate = 0.09, loss = 'ls')
model.fit(x_train,y_train)
score = model.score(x_test,y_test)*100
print("Model Accuracy =",score,"%")