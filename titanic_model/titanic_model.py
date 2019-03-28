import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.filterwarnings('ignore')

df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
def features(x):
    x['FamilySize'] = x['SibSp'] + x['Parch'] + 1
    x['IsAlone'] = pd.Series()
    x['IsAlone'][x['FamilySize'] == 1] = 1
    x['IsAlone'][x['FamilySize'] > 1] = 0
    x['FamilyName'] = x['Name'].str.split(',').str[0]
    x['Title'] = x['Name'].str.split(',').str[1].str.split('.').str[0]
    x['Embarked'].fillna('S',inplace=True)
    x['Age'].fillna(x['Age'].median(),inplace=True)
    x['Cabin'][x['Cabin'].isna()] = 0
    x['Cabin'][x['Cabin'].notna()] = 1
    x['Title'] = x['Title'].replace(['Rev','Don','Mme','Lady','Jonkheer','Mlle','the Count','Major','Capt','Master'],'Rare')
    x['Fare'] = x['Fare'].fillna(x['Fare'].median())
    return x
def vector(x):
    encoder = LabelEncoder()
    encoder2 = OneHotEncoder()
    age_range = [0,13,18,34,50,100]
    age_label = [1,2,3,4,5]
    fare_range = [0,129,258,387,516]
    fare_label = [1,2,3,4]
    sib_range = [0,2,4,6,8]
    sib_label = [1,2,3,4]
    parch_range = [0,3,6,9]
    parch_label = [1,2,3]
    x['SibSp'] = pd.cut(x['SibSp'],sib_range,labels=sib_label,include_lowest=True)
    x['Parch'] = pd.cut(x['Parch'],parch_range,labels=parch_label,include_lowest=True)
    x['Fare'] = pd.cut(x['Fare'],fare_range,labels=fare_label,include_lowest = True)
    x['Age'] = pd.cut(x['Age'],age_range,labels=age_label)
    x['Sex'] = encoder.fit_transform(x['Sex'])
    x['Embarked'] = encoder.fit_transform(x['Embarked'])
    x['Title'] = encoder2.fit_transform(x[['Title']]).toarray()
    x['FamilyName'] = encoder2.fit_transform(x[['FamilyName']]).toarray()
    return x
df_train = features(df_train)
df_test = features(df_test)
df_train = vector(df_train)
df_test = vector(df_test)
x_train = df_train.drop(['PassengerId','Survived','Ticket','Name'],axis=1)
y_train = df_train.Survived
x_test = df_test.drop(['PassengerId','Ticket','Name'],axis=1)
model = RandomForestClassifier(random_state=100,max_features='sqrt')
model.fit(x_train,y_train)
y_test = model.predict(x_test)
y_predicted = pd.DataFrame()
y_predicted['PassengerId'] = df_test['PassengerId']
y_predicted['Survived'] = pd.DataFrame(y_test)
y_predicted.set_index('PassengerId',inplace=True)
y_predicted.to_csv('gender_submission2.csv')
