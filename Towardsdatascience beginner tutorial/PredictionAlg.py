# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 13:22:58 2019

Predictive model implementation testing without validation.
Following the guidance of: https://towardsdatascience.com/how-to-begin-your-own-data-science-journey-2223caad8cee

@author: dlassog
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression 

train = pd.read_csv('train.csv')
train['Sex'] = train['Sex'].apply(lambda sex:1 if sex=='male' else 0)
train['Age'] = train['Age'].fillna(train['Age'].mean())
train['Fare'] = train['Fare'].fillna(train['Age'].mean())

test = pd.read_csv('test.csv')
test['Sex'] = test['Sex'].apply(lambda sex:1 if sex=='male' else 0)
test['Age'] = test['Age'].fillna(test['Age'].mean())
test['Fare'] = test['Fare'].fillna(test['Age'].mean())

survived = train['Survived'].values
cols=["Pclass","Sex","Age"]
data_train = train[cols].values
data_test = test[cols].values

model = LogisticRegression()
model.fit(data_train,survived)
predict = model.predict(data_test)