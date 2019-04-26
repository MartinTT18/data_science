#creating model for predicting weather the mushroom is edible or poisonous
#import all the modules needed for the code
import pandas as pd
import numpy as np
import math
import random
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#load the file
df = pd.read_csv('mushrooms.csv')
#create the x file
df_x = df.drop('class',axis=1)
#create the y file
df_y = df['class']
#define the encoders
label = LabelEncoder()
hot = OneHotEncoder()
#encode the string data to numerical data
df_y = pd.DataFrame(label.fit_transform(df_y))
df_encoded = hot.fit_transform(df_x)
#split the data to test and train data
x_train,x_test,y_train,y_test = train_test_split(df_encoded,df_y,test_size=0.2)
#define the classifier algorithm  and predict the data
model_1 = KNeighborsClassifier()
model_1.fit(x_train,y_train)
y_predicted = model_1.predict(x_test)
#get the accuracy of the model and print it
error = accuracy_score(y_test,y_predicted)
print(error)
