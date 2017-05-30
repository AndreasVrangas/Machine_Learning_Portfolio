# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 21:42:33 2017

@author: Riko
"""

import pandas as pd
import numpy as np #Linear Algebra
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

#Import Data
df=pd.read_csv('mushrooms.csv')

plt.subplot(3,1,1)
sns.countplot(x='cap-shape',hue='class',data=df)
plt.title('class per shape')
plt.subplot(3,1,2)
sns.countplot(x='cap-surface',hue='class',data=df)
plt.title('class per surface')
plt.subplot(3,1,3)
sns.countplot(x='cap-color',hue='class',data=df)
plt.title('class per color')
#plt.tight_layout()
plt.show()

plt.subplot(4,1,1)
sns.countplot(x='gill-attachment',hue='class',data=df)
plt.title('class per attachmnt')
plt.subplot(4,1,2)
sns.countplot(x='gill-spacing',hue='class',data=df)
plt.title('class per spacing')
plt.subplot(4,1,3)
sns.countplot(x='gill-size',hue='class',data=df)
plt.title('class per size')
plt.subplot(4,1,4)
sns.countplot(x='gill-color',hue='class',data=df)
plt.title('class per color')
#plt.tight_layout()
plt.show()



plt.subplot(6,2,1)
sns.countplot(x='stalk-shape',hue='class',data=df)
plt.title('class per shape')
plt.subplot(6,2,2)
sns.countplot(x='stalk-root',hue='class',data=df)
plt.title('class per root')
plt.subplot(6,2,3)
sns.countplot(x='stalk-surface-above-ring',hue='class',data=df)
plt.title('class per surface above ring')
plt.subplot(6,2,4)
sns.countplot(x='stalk-surface-below-ring',hue='class',data=df)
plt.title('class per surface bellow ring')
plt.subplot(6,2,5)
sns.countplot(x='stalk-color-above-ring',hue='class',data=df)
plt.title('class per color above ring')
plt.subplot(6,2,6)
sns.countplot(x='stalk-color-below-ring',hue='class',data=df)
plt.title('class per color bellow ring')
#plt.tight_layout()
plt.show()


plt.subplot(2,1,1)
sns.countplot(x='veil-type',hue='class',data=df)
plt.title('class per veil type')
plt.subplot(2,1,2)
sns.countplot(x='veil-color',hue='class',data=df)
plt.title('class per veil color')
#plt.tight_layout()
plt.show()

plt.subplot(2,1,1)
sns.countplot(x='ring-number',hue='class',data=df)
plt.title('class per ring number')
plt.subplot(2,1,2)
sns.countplot(x='ring-type',hue='class',data=df)
#plt.title('class per ring type')
plt.show()



sns.countplot(x='gill-color',hue='ring-type',data=df)

plt.subplot(5,1,1)
sns.countplot(x='bruises',hue='class',data=df)
plt.title('Class per bruises')
plt.subplot(5,1,2)
sns.countplot(x='odor',hue='class',data=df)
plt.title('Class per odor')
plt.subplot(5,1,3)
sns.countplot(x='spore-print-color',hue='class',data=df)
plt.title('Class per SPC')
plt.subplot(5,1,4)
sns.countplot(x='population',hue='class',data=df)
plt.title('Class per population')
plt.subplot(5,1,5)
sns.countplot(x='habitat',hue='class',data=df)
plt.title('Class per habitat')
plt.tight_layout()
plt.show()

#Encoding!!(Same thing, different way for bruises and for class)
df['bruises'] = np.where(df['bruises'] == 't',1,0)
df['class'] = np.where(df['class'] == 'p',0,1)


#############################################Feature Engineering#############################################

#########Based on Gill Subplots
df['Free_Buffs'] = np.where(((df['gill-attachment']=='f')&(df['gill-color']=='b')),1,0)
sns.countplot(x='Free_Buffs',hue='class',data=df)
#########Based on Ring Subplots
df['one-large']=np.where(((df['ring-number']=='o')&(df['ring-type']=='l')),1,0)
sns.countplot(x='one-large',hue='class',data=df)
#########Based on the rest Subplots
df['hab-odor'] = np.where(((df['habitat']=='d')&((df['odor']=='f')|(df['odor']=='y')|(df['odor']=='s'))),1,0)
#########Based on Stalk Subplots
df['color-above-below']=np.where(((df['stalk-color-above-ring']== 'w')&(df['stalk-color-below-ring']=='g')),1,0)

df['ppl-hap']=np.where(((df['population']=='n')|(df['population']=='a'))&((df['habitat']=='g')|(df['habitat']=='d')),1,0)

train=df.sample(frac=0.8,random_state=150)
test=df.drop(train.index)
Y=train['class']
true_cl=test['class']

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import cross_validation,metrics
from sklearn.linear_model import LogisticRegression
#1st train set: made variables vs class
train_1=train[["Free_Buffs","one-large","ppl-hap"]]
test_1=test[["Free_Buffs","one-large","ppl-hap"]]

X_train_1,X_test_1,y_train_1,y_test_1 = cross_validation.train_test_split(train_1,Y,test_size=0.2,random_state=100)
rf=RandomForestClassifier()
rf=rf.fit(X_train_1,y_train_1)
confidence_1_rf=rf.score(X_test_1,y_test_1)
pred_1_rf=rf.predict(test_1)
accuracy_1_rf=metrics.accuracy_score(true_cl,rf.predict(test_1))

print ("Cross-validation score of feature engineered variables [Random Forest]:%.2f" %confidence_1_rf)
print("Accuracy score of feature engineered variables [Random Forest] :f", round(accuracy_1_rf*100,2),'%')
print("Feature engineered variables confusion matrix [Random Forest]:\n", metrics.confusion_matrix(true_cl,pred_1_rf))
print("AUC score for feature engineered variables [Random Forest]: %.2f", metrics.roc_auc_score(true_cl,pred_1_rf))
print("[Random Forest] classification report:\n",metrics.classification_report(true_cl,pred_1_rf))



nb=LogisticRegression()
nb=nb.fit(X_train_1,y_train_1)
confidence_1_nb=nb.score(X_test_1,y_test_1)
pred_1_nb=nb.predict(test_1)
accuracy_1_nb=metrics.accuracy_score(true_cl,nb.predict(test_1))

print ("Cross-validation score of feature engineered variables [Naive Bayes]:%.2f" %confidence_1_nb)
print("Accuracy score of feature engineered variables [Naive Bayes] :%.2f" %accuracy_1_nb)
print("Feature engineered variables confusion matrix [Naive Bayes]:\n", metrics.confusion_matrix(true_cl,pred_1_nb))
print("AUC score for feature engineered variables [Naive Bayes]: %.2f", metrics.roc_auc_score(true_cl,pred_1_nb))
print("[Naive Bayes] classification report:\n",metrics.classification_report(true_cl,pred_1_nb))

train_2 = train[["color-above-below","Free_Buffs","one-large"]]
test_2 = test[["color-above-below","Free_Buffs","one-large"]]
 
X_train_2,X_test_2,y_train_2,y_test_2 = cross_validation.train_test_split(train_2,Y,test_size=0.2,random_state=100)
rf=RandomForestClassifier()
rf=rf.fit(X_train_2,y_train_2)
confidence_2_rf=rf.score(X_test_2,y_test_2)
pred_2_rf=rf.predict(test_2)
accuracy_2_rf=metrics.accuracy_score(true_cl,rf.predict(test_2))
print ("Cross-validation score of  [Random Forest]:%.2f" %confidence_2_rf)
print("Accuracy score of [Random Forest] :%.2f" %accuracy_2_rf)
print("[Random Forest] confusion matrix:\n" ,metrics.confusion_matrix(true_cl,pred_2_rf))
print("AUC score for [Random Forest]: %.2f" %metrics.roc_auc_score(true_cl,pred_2_rf))
print(metrics.classification_report(true_cl,pred_2_rf))

nb=nb.fit(X_train_2,y_train_2)
confidence_2_nb=nb.score(X_test_2,y_test_2)
pred_2_nb=nb.predict(test_2)
accuracy_2_nb=metrics.accuracy_score(true_cl,nb.predict(test_2))
print ("Cross-validation score of  [Naive Bayes]:%.2f" %confidence_2_nb)
print("Accuracy score of [[Naive Bayes]] :%.2f" %accuracy_2_nb)
print("[[Naive Bayes]] confusion matrix:\n" ,metrics.confusion_matrix(true_cl,pred_2_nb))
print("AUC score for [Naive Bayes]]:%.2f", metrics.roc_auc_score(true_cl,pred_2_nb))
print("[Naive Bayes] classification report:\n",metrics.classification_report(true_cl,pred_2_nb))