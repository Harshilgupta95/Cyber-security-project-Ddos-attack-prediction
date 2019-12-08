import time
startTime = time.time()
from warnings import simplefilter
import matplotlib.pyplot as plt

simplefilter(action='ignore', category=FutureWarning)
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
import numpy as np
import operator
class EnsembleClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, clfs, weights=None):
        self.clfs = clfs
        self.weights = weights

    def fit(self, X, y):

        for clf in self.clfs:
            clf.fit(X, y)

    def predict(self, X):


        self.classes_ = np.asarray([clf.predict(X) for clf in self.clfs])
        if self.weights:
            avg = self.predict_proba(X)

            maj = np.apply_along_axis(lambda x: max(enumerate(x), key=operator.itemgetter(1))[0], axis=1, arr=avg)

        else:
            maj = np.asarray([np.argmax(np.bincount(self.classes_[: ,c])) for c in range(self.classes_.shape[1])])

        return maj

    def predict_proba(self, X):


        self.probas_ = [clf.predict_proba(X) for clf in self.clfs]
        avg = np.average(self.probas_, axis=0, weights=self.weights)

        return avg


from sklearn import datasets
print('1.UDP\n2.TCPSYN\n3.ICMP\n4.LAND\n5.TCPSYNACK')
attack=input('Attack name: ')
path='C:\\Users\\LENOVO\\Downloads\\'+attack+'.csv'
import pandas as pd
dataset = pd.read_csv(path,low_memory=False)

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
dataset['UTC Time'] = le.fit_transform(dataset['UTC Time'])
dataset['Absolute Time'] = le.fit_transform(dataset['Absolute Time'])
dataset['Source'] = le.fit_transform(dataset['Source'])
dataset['Destination'] = le.fit_transform(dataset['Destination'])
dataset['Protocol'] = le.fit_transform(dataset['Protocol'])
dataset['SourcePort'] = le.fit_transform(dataset['SourcePort'])
dataset['DestPort'] = le.fit_transform(dataset['DestPort'])
dataset['Hwdestaddr'] = le.fit_transform(dataset['Hwdestaddr'])
dataset['Hwsrcaddr'] = le.fit_transform(dataset['Hwsrcaddr'])
dataset['Unresolved Destport'] = le.fit_transform(dataset['Unresolved Destport'])
dataset['Unresolved Srcport'] = le.fit_transform(dataset['Unresolved Srcport'])

A = ['Delta Time' ,'Length' ,'Cumulative Bytes' ,'Time' ,'UTC Time' ,'Absolute Time' ,'Source' ,'Destination'
     ,'Protocol' ,'SourcePort' ,'DestPort' ,'Hwsrcaddr' ,'Hwdestaddr' ,'Unresolved Destport' ,'Unresolved Srcport']
X = dataset[A]
y = dataset['Class']
y= le.fit_transform(y)
from sklearn import model_selection
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import linear_model

from sklearn.ensemble import RandomForestClassifier
import numpy as np
import time as ti
from sklearn.svm import SVC

np.random.seed(123)
clf3 = LogisticRegression()
# clf2 = linear_model.SGDClassifier()
clf2 = DecisionTreeClassifier()
clf1 = AdaBoostClassifier()

clf4=RandomForestClassifier(n_estimators=1000,max_depth=200,random_state=1)
clf6=GaussianNB()
clf7=SVC(kernel='rbf',probability=True)


df = pd.DataFrame(columns=('w1', 'w2', 'w3', 'mean', 'std'))
eclf = EnsembleClassifier(clfs=[clf1, clf2, clf3], weights=[1, 1, 2])
i = 0
for clf, label in zip([clf1, clf2, clf3, eclf],
                      ['AdaBoost', 'DecesionTree', 'LogisticRegression','WeightedEnsembleClassifier']):
    scores = model_selection.cross_val_score(clf, X, y, cv=5, scoring='accuracy')

    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))

from sklearn.model_selection import cross_val_predict
predictions = cross_val_predict(eclf, X, y)

import scikitplot as skplt
skplt.metrics.plot_confusion_matrix(y, predictions, normalize=True)
plt.show()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,random_state=42)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


from Stacking import EnsembleClassifier
np.random.seed(123)
eclf = EnsembleClassifier(clfs=[clf1,clf2,clf3], weights=[1,1,1])
eclf1= EnsembleClassifier(clfs=[clf1,clf2,clf3], weights=[1,2,3])

from sklearn.model_selection import cross_val_predict
predictions = cross_val_predict(eclf, X, y)
predictions1 = cross_val_predict(eclf1, X, y)

import scikitplot as skplt
skplt.metrics.plot_confusion_matrix(y, predictions, normalize=True)
plt.show()
skplt.metrics.plot_confusion_matrix(y, predictions1, normalize=True)
plt.show()
