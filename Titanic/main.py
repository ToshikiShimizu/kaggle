# coding: utf-8
"""
very simple implementation with XGBoost
"""
import pandas as pd
import numpy as np
#Load data
train = pd.read_csv("../input/train.csv",header=0)
test = pd.read_csv("../input/test.csv",header=0)
#Fill the missing values
train['Fare'] = train['Fare'].fillna(train['Fare'].median())
test['Fare'] = test['Fare'].fillna(train['Fare'].median())
train['Embarked'] = train['Embarked'].fillna('S')# S is the most common symbol
test['Embarked'] = test['Embarked'].fillna('S')
#Calc number of family
train['family'] = train['SibSp'] + train['Parch']
test['family'] = test['SibSp'] + test['Parch']
#Convert categorical variable into dummy/indicator variables
def add_dummy(df):
    df['Pclass'] = df['Pclass'].astype(np.str)
    temp = pd.get_dummies(df[['Sex','Embarked','Pclass']], drop_first = False)
    temp['PassengerId'] = df['PassengerId']
    return pd.merge(df, temp)
train = add_dummy(train)
test = add_dummy(test)
#Drop unnecessary feature
def get_feature_mat(df):
    temp = df.drop(columns=['PassengerId','Name','Sex','SibSp','Parch','Ticket','Embarked','Age','Cabin','Pclass'])
    try:
        temp = temp.drop(columns=['Survived'])
    except:
        pass
    print (temp)
    return temp.as_matrix()
x_train = get_feature_mat(train)
x_test = get_feature_mat(test)
y_train = train['Survived'].as_matrix()
#Fit and Predict
from xgboost import XGBClassifier
xgb = XGBClassifier(max_depth=3, learning_rate=0.1, n_estimators=100)
xgb.fit(x_train, y_train)
y_test_xgb = xgb.predict(x_test)
#Generate timestamp
from datetime import datetime, timedelta, timezone
JST = timezone(timedelta(hours=+9), 'JST')
ts = datetime.now(JST).strftime('%y%m%d%H%M')
#Save
test["Survived"] = y_test_xgb.astype(np.int)
test[["PassengerId","Survived"]].to_csv(('submit_'+ts+'.csv'),index=False)
test["Survived"].head()
