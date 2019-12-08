import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from Stacking import Stacking
from sklearn.model_selection import train_test_split


print('1.UDP\n2.TCPSYN\n3.ICMP\n4.LAND\n5.TCPSYNACK')
attack=input('Attack name: ')
path='C:\\Users\\LENOVO\\Downloads\\'+attack+'.csv'
df = pd.read_csv(path,sep=",",usecols=(9,11,12,13,16,21))

# df = pd.read_csv("train.csv")
# print(df)

# In[4]:

le = LabelEncoder()
# le.fit(df.type)
y = le.transform(df[21])


# In[7]:

# df.drop('id',1,inplace=True)
# df.drop('color',1,inplace=True)
# df.drop('type',1,inplace=True)


# In[11]:




# In[14]:

x = np.array(df,dtype=float)
print(x)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
estimators = [LogisticRegression(C=0.8),RandomForestClassifier(n_estimators=500)]
stack_model = LogisticRegression()
stk = Stacking(estimators,stack_model,use_prob=False,n_splits=5,verbose=1)
stk.fit(X_train,y_train)
y_pred = stk.predict(X_test)

from sklearn.metrics import classification_report,f1_score,accuracy_score,confusion_matrix

print ("class rep",classification_report(y_test,y_pred))
print ("confusion_matrix",confusion_matrix(y_test,y_pred))
print ("accuracy_score",accuracy_score(y_test,y_pred))