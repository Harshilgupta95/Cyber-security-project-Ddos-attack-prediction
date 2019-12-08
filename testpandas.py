import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import numpy as np
le = LabelEncoder()
ohe= OneHotEncoder()

print('1.UDP\n2.TCPSYN\n3.ICMP\n4.LAND\n5.TCPSYNACK')
attack=input('Attack name: ')
path='C:\\Users\\LENOVO\\Downloads\\'+attack+'.csv'
io = pd.read_csv(path,sep=",",usecols=(9,11,12,13,16,21))


io.head()

train=io.sample(frac=0.8,random_state=200)
test=io.drop(train.index)

feature_names=['Length','Cumulative Bytes']
X = io[feature_names]
y = io['Class']

#Create training and test dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,random_state=42)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


#Classifier Gaussian Naive Bayes

# from sklearn.naive_bayes import GaussianNB
# gnb = GaussianNB()
# gnb.fit(X_train, y_train)


#Logistic Regression Classification

# from sklearn.linear_model import LogisticRegression
# logreg = LogisticRegression()
# logreg.fit(X_train, y_train)


#Random Forest Classifier

# from sklearn.ensemble import RandomForestClassifier
# rf = RandomForestClassifier()
# rf.fit(X_train, y_train)

#Decesion Tree
from sklearn import tree
dt = tree.DecisionTreeClassifier()
dt.fit(X_train, y_train)
print('\nDecesionTree: {}'.format(dt.score(X_test, y_test)))


#Gradient Boosting
from sklearn.ensemble import GradientBoostingClassifier
learn=[0.2,0.5,1.0,1.5,2.0]
for i in learn:
    gb_clf = GradientBoostingClassifier(n_estimators=20, learning_rate=i, max_features=2, max_depth=2, random_state=0)
    gb_clf.fit(X_train, y_train)
    print('Gradient Boosting: {}'.format(gb_clf.score(X_test, y_test)) + " Learning Rate: {}".format(i))

#Guassian Classifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
k=[0.5,1.0,1.5,2.0]
for i in k:
    kernel = i * RBF(length_scale=1.0)
    gpc = GaussianProcessClassifier(kernel=kernel)
    gpc.fit(X_train[:100], y_train[:100])
    print('Gaussian Classifier: {}'.format(gpc.score(X_test, y_test))+" Kernel: {}".format(i))



#Comparision

# print('gnb: {}'.format(gnb.score(X_test, y_test)))
# print('rf: {}'.format(rf.score(X_test, y_test)))
# print('log_reg: {}'.format(logreg.score(X_test, y_test)))
# print('DecesionTree: {}'.format(dt.score(X_test, y_test)))
# print('Gradient Boosting: {}'.format(gb_clf.score(X_test, y_test)))
# print('Gaussian Classifier: {}'.format(gpc.score(X_test, y_test)))


#Ensemble Classifier

from sklearn.ensemble import VotingClassifier
# estimators=[('gnb', gnb), ('rf', rf), ('log_reg', logreg),('decesiontree',dt),('gradientBoost',gb_clf),('gaussian',gpc)]
estimators=[('decesiontree',dt),('gradientBoost',gb_clf),('gaussian',gpc)]
ensemble = VotingClassifier(estimators, voting='hard')
ensemble.fit(X_train[:100], y_train[:100])
print('Ensemble: '+str(ensemble.score(X_test, y_test))+"\n")

# y_pred_class1 = gnb.predict(X_test)
# y_pred_class2= rf.predict(X_test)
# y_pred_class3 = logreg.predict(X_test)
y_pred_class4 =dt.predict(X_test)
y_pred_class5 =gb_clf.predict(X_test)
y_pred_class6 =gpc.predict(X_test)


y_test_le = le.fit_transform(y_test)
# y_pred_class1_le = le.fit_transform(y_pred_class1)
# y_pred_class2_le = le.fit_transform(y_pred_class2)
# y_pred_class3_le = le.fit_transform(y_pred_class3)
y_pred_class4_le = le.fit_transform(y_pred_class4)
y_pred_class5_le = le.fit_transform(y_pred_class5)
y_pred_class6_le = le.fit_transform(y_pred_class6)

#GNB

class1_tp = 0
class1_fn = 0
class1_fp = 0
class1_tn = 0

# print("For GNB: ")
# for i in range(0,len(y_pred_class1)):
#      if( y_test_le[i] == y_pred_class1_le[i]):
#          class1_tp +=1
#      if( y_test_le[i] != y_pred_class1_le[i]):
#          class1_fn += 1
#      if(y_pred_class1_le[i]==1 and y_test_le[i]!=y_pred_class1_le[i]):
#           class1_fp+=1
#      if(y_test_le[i]==y_pred_class1_le[i]==0):
#           class1_tn+=1
#
# recall=class1_tp/(class1_tp+class1_fn)
# precision=class1_tp/(class1_tp+class1_fp)
# f1=2*((recall*precision)/(recall+precision))
#
# print("\nTP:" + str(class1_tp))
# print("FN:" + str(class1_fn))
# print("FP:" + str(class1_fp))
# print("TN:" + str(class1_tn))
# print("Recall :" + str(recall))
# print("Precision :" + str(precision))
# print("F-measure :" + str(f1)+"\n")
#
# print("For RF: ")
# for i in range(0,len(y_pred_class2)):
#      if( y_test_le[i] == y_pred_class2_le[i]):
#          class1_tp +=1
#      if( y_test_le[i] != y_pred_class2_le[i]):
#          class1_fn += 1
#      if(y_pred_class2_le[i]==1 and y_test_le[i]!=y_pred_class2_le[i]):
#           class1_fp+=1
#      if(y_test_le[i]==y_pred_class2_le[i]==0):
#           class1_tn+=1
#
# recall=class1_tp/(class1_tp+class1_fn)
# precision=class1_tp/(class1_tp+class1_fp)
#
# f1=2*((recall*precision)/(recall+precision))
#
# print("\nTP:" + str(class1_tp))
# print("FN:" + str(class1_fn))
# print("FP:" + str(class1_fp))
# print("TN:" + str(class1_tn))
# print("Recall :" + str(recall))
# print("Precision :" + str(precision))
# print("F-measure :" + str(f1)+"\n")
#
#
#
# print("For Log_Reg: ")
# for i in range(0,len(y_pred_class3)):
#      if( y_test_le[i] == y_pred_class3_le[i]):
#          class1_tp +=1
#      if( y_test_le[i] != y_pred_class3_le[i]):
#          class1_fn += 1
#      if(y_pred_class3_le[i]==1 and y_test_le[i]!=y_pred_class3_le[i]):
#           class1_fp+=1
#      if(y_test_le[i]==y_pred_class3_le[i]==0):
#           class1_tn+=1
#
# recall=class1_tp/(class1_tp+class1_fn)
# precision=class1_tp/(class1_tp+class1_fp)
# f1=2*((recall*precision)/(recall+precision))
#
# print("\nTP:" + str(class1_tp))
# print("FN:" + str(class1_fn))
# print("FP:" + str(class1_fp))
# print("TN:" + str(class1_tn))
# print("Recall :" + str(recall))
# print("Precision :" + str(precision))
# print("F-measure :" + str(f1)+"\n")

print("For Decesion Tree: ")
for i in range(0,len(y_pred_class4)):
     if( y_test_le[i] == y_pred_class4_le[i]):
         class1_tp +=1
     if( y_test_le[i] != y_pred_class4_le[i]):
         class1_fn += 1
     if(y_pred_class4_le[i]==1 and y_test_le[i]!=y_pred_class4_le[i]):
          class1_fp+=1
     if(y_test_le[i]==y_pred_class4_le[i]==0):
          class1_tn+=1

recall=class1_tp/(class1_tp+class1_fn)
precision=class1_tp/(class1_tp+class1_fp)
f1=2*((recall*precision)/(recall+precision))

print("\nTP:" + str(class1_tp))
print("FN:" + str(class1_fn))
print("FP:" + str(class1_fp))
print("TN:" + str(class1_tn))
print("Recall :" + str(recall))
print("Precision :" + str(precision))
print("F-measure :" + str(f1)+"\n")

print("For Gradient Boosting: ")
for i in range(0,len(y_pred_class5)):
     if( y_test_le[i] == y_pred_class5_le[i]):
         class1_tp +=1
     if( y_test_le[i] != y_pred_class5_le[i]):
         class1_fn += 1
     if(y_pred_class5_le[i]==1 and y_test_le[i]!=y_pred_class5_le[i]):
          class1_fp+=1
     if(y_test_le[i]==y_pred_class5_le[i]==0):
          class1_tn+=1

recall=class1_tp/(class1_tp+class1_fn)
precision=class1_tp/(class1_tp+class1_fp)
f1=2*((recall*precision)/(recall+precision))

print("\nTP:" + str(class1_tp))
print("FN:" + str(class1_fn))
print("FP:" + str(class1_fp))
print("TN:" + str(class1_tn))
print("Recall :" + str(recall))
print("Precision :" + str(precision))
print("F-measure :" + str(f1)+"\n")

print("For Guassian: ")
for i in range(0,len(y_pred_class6)):
     if( y_test_le[i] == y_pred_class6_le[i]):
         class1_tp +=1
     if( y_test_le[i] != y_pred_class6_le[i]):
         class1_fn += 1
     if(y_pred_class6_le[i]==1 and y_test_le[i]!=y_pred_class6_le[i]):
          class1_fp+=1
     if(y_test_le[i]==y_pred_class6_le[i]==0):
          class1_tn+=1

recall=class1_tp/(class1_tp+class1_fn)
precision=class1_tp/(class1_tp+class1_fp)
f1=2*((recall*precision)/(recall+precision))

print("\nTP:" + str(class1_tp))
print("FN:" + str(class1_fn))
print("FP:" + str(class1_fp))
print("TN:" + str(class1_tn))
print("Recall :" + str(recall))
print("Precision :" + str(precision))
print("F-measure :" + str(f1)+"\n")
