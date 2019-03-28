import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.metrics import accuracy_score
import warnings

df = pd.read_csv('Iris.csv')
encoder = LabelEncoder()
df['Species'] = encoder.fit_transform(df['Species'])
x_train,x_test,y_train,y_test = train_test_split(df.drop(['Species'],axis=1),df['Species'],test_size=0.2)
model = LogisticRegression()
model.fit(x_train,y_train)
y_predict = model.predict(x_test)
score = accuracy_score(y_test,y_predict)
print(score)
