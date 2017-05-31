# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 11:24:05 2016

@author: Riko
"""
import pandas as pd
import numpy as np
import math
from sklearn.linear_model import LinearRegression,LogisticRegression
import matplotlib.pyplot as plt
from sklearn import cross_validation, svm, metrics
from sklearn.preprocessing import scale
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA

'Import Dataset'
names=['Relative Compactness','Surface Area','Wall Area','Roof Area','Overall Height','Orientation','Glazing Area','Glazing Area Distribution','Heating Load','Cooling Load']
df = pd.read_excel('ENB2012_data.xlsx',names=names)

#writer = pd.ExcelWriter('Energy.xlsx')
#Fill the file with the data
#df.to_excel(writer,'Sheet1')
#writer.save()
#Save the file'''
'Split Dataset into Train and Test set (test set considered to be 20% of the observations)'

'''corr_matrix=df.corr()
writer=pd.ExcelWriter('corr_coef.xlsx')
corr_matrix.to_excel(writer,'Sheet1')
writer.save()'''

train=df.sample(frac=0.8,random_state=150)
test=df.drop(train.index)

#train=df[-610:]
#test=df[:-610
test_loads=test[["Cooling Load"]]

   

'Drop the Load values and add Nan values instead'
#test.drop(['Heating Load', 'Cooling Load'],axis=1,inplace=True)
#test['Heat_label']=0
#test['Cooling_label']=0



df.dropna(inplace=True)

Y1=np.array(train['Heating Load'])

Y2=np.array(train['Cooling Load'])




train_5=train[['Overall Height','Relative Compactness','Roof Area','Surface Area']]
test_5=test[['Overall Height','Relative Compactness','Roof Area','Surface Area']]

X_train,X_test,y_train,y_test=cross_validation.train_test_split(train_5,Y2,test_size=0.2)
clf=RandomForestRegressor(random_state=100)
clf.fit(X_train,y_train)
confidence_5= clf.score(X_test,y_test)
prediction_all_four=clf.predict(test_5)
print("Confidence for all four:%.2f" %confidence_5)
print("R^2 score for all four:%.2f" %metrics.r2_score(test_loads['Cooling Load'],prediction_all_four))
print("Feature Importance for all four:",clf.feature_importances_)
plt.plot(np.linspace(0,153,154),test_loads,color='black')
plt.plot(prediction_all_four,color='blue',linewidth=1.5)
plt.ylabel("KWh")
plt.legend(['true load','prediction'], loc=2)
plt.show()



#Positivie Correlation
train_6=train[['Overall Height','Relative Compactness']]
test_6=test[['Overall Height','Relative Compactness']]

X_train,X_test,y_train,y_test=cross_validation.train_test_split(train_6,Y2,test_size=0.2)
clf=RandomForestRegressor(random_state=100)
clf.fit(X_train,y_train)
confidence_6= clf.score(X_test,y_test)
prediction_positive=clf.predict(test_6)
print("Confidence for positives:%.2f" %confidence_6)
print("R^2 score for positives:%.2f" %metrics.r2_score(test_loads['Cooling Load'],prediction_positive))
print("Feature Importance for positives: ",clf.feature_importances_)
plt.plot(np.linspace(0,153,154),test_loads,color='black')
plt.plot(prediction_positive,color='blue',linewidth=1.5)
plt.ylabel("KWh")
plt.legend(['true load','prediction'], loc=2)
plt.show()




#Negative Correlation!!
train_7=train[['Roof Area','Surface Area']]
test_7=test[['Roof Area','Surface Area']]

X_train,X_test,y_train,y_test=cross_validation.train_test_split(train_7,Y2,test_size=0.2)
clf=RandomForestRegressor(random_state=100)
clf.fit(X_train,y_train)
confidence_7= clf.score(X_test,y_test)
prediction_negative=clf.predict(test_7)
print("Confidence for negatives:%.2f" %confidence_7)
print("R^2 score for negatives:%.2f" %metrics.r2_score(test_loads,prediction_negative))
print("Feature Importance for negatives:",clf.feature_importances_)
plt.plot(np.linspace(0,153,154),test_loads,color='black')
plt.plot(prediction_negative,color='blue',linewidth=1.5)
plt.ylabel("KWh")
plt.legend(['true load','prediction'], loc=2)
plt.show()


train_8=train[['Overall Height','Relative Compactness','Roof Area','Surface Area','Wall Area','Glazing Area Distribution','Glazing Area','Orientation']]
test_8=test[['Overall Height','Relative Compactness','Roof Area','Surface Area','Wall Area','Glazing Area Distribution','Glazing Area','Orientation']]

X_train,X_test,y_train,y_test=cross_validation.train_test_split(train_8,Y2,test_size=0.2)
clf=RandomForestRegressor(random_state=100)
clf.fit(X_train,y_train)
confidence_8= clf.score(X_test,y_test)
prediction_8=clf.predict(test_8)
print("Confidence for all features:%.2f" %confidence_8)
print("R^2 score for all features:%.2f" %metrics.r2_score(test_loads,prediction_8))
print("Feature Importance for all features:", clf.feature_importances_)
plt.plot(np.linspace(0,153,154),test_loads,color='black')
plt.plot(prediction_8,color='blue',linewidth=1.5)
plt.ylabel("KWh")
plt.legend(['true load','prediction'], loc=2)
plt.show()

train_9=train[['Glazing Area','Wall Area','Glazing Area Distribution','Orientation']]
test_9=test[['Glazing Area','Wall Area','Glazing Area Distribution','Orientation']]

X_train,X_test,y_train,y_test=cross_validation.train_test_split(train_9,Y2,test_size=0.2)
clf=RandomForestRegressor(random_state=100)
clf.fit(X_train,y_train)
confidence_9= clf.score(X_test,y_test)
prediction_9=clf.predict(test_9)
print("Confidence for uncorrelated:%.2f" %confidence_9)
print("R^2 score for uncorrelated:%.2f" %metrics.r2_score(test_loads,prediction_9))
print("Feature Importance for uncorrelated:",clf.feature_importances_)
plt.plot(np.linspace(0,153,154),test_loads,color='black')
plt.plot(prediction_9,color='blue',linewidth=1.5)
plt.ylabel("KWh")
plt.legend(['true load','prediction'], loc=2)
plt.show()

