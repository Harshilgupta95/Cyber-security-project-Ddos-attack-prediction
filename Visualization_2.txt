import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
# X, y = load_digits(return_X_y=True)
# from sklearn.ensemble import RandomForestClassifier
# random_forest_clf = RandomForestClassifier(n_estimators=5, max_depth=5, random_state=1)

import pandas as pd


print('1.UDP\n2.TCPSYN\n3.ICMP\n4.LAND\n5.TCPSYNACK')
attack=input('Attack name: ')
path='C:\\Users\\LENOVO\\Downloads\\'+attack+'.csv'
io = pd.read_csv(path,sep=",",usecols=(9,11,12,13,16,21))

feature_names=['Length','Cumulative Bytes']
X = io[feature_names]
# y = io['Class']

io['Class'].unique()
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()

y = label_encoder.fit_transform(io['Class'])



from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from mlxtend.classifier import StackingCVClassifier
import numpy as np
import warnings
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import tree

warnings.simplefilter('ignore')

RANDOM_SEED = 42

clf1 = KNeighborsClassifier(n_neighbors=1)
clf2 = RandomForestClassifier(random_state=RANDOM_SEED)
clf3 = GaussianNB()
clf4=tree.DecisionTreeClassifier()
lr = LogisticRegression()
gb_clf = GradientBoostingClassifier(n_estimators=20, learning_rate=20, max_features=2, max_depth=2, random_state=0)
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
kernel = 1.0 * RBF(length_scale=1.0)
gpc = GaussianProcessClassifier(kernel=kernel)


# Starting from v0.16.0, StackingCVRegressor supports
# `random_state` to get deterministic result.
sclf = StackingCVClassifier(classifiers=[clf4, clf2, gb_clf],
                            meta_classifier=lr,
                            random_state=RANDOM_SEED)




from sklearn.model_selection import cross_val_predict
predictions = cross_val_predict(sclf, X, y)

import scikitplot as skplt
skplt.metrics.plot_confusion_matrix(y, predictions, normalize=True)
plt.show()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,random_state=42)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.ensemble import VotingClassifier
estimators=[('dtree',clf4),('gbc',gb_clf),('rf',clf2)]
ensemble = VotingClassifier(estimators, voting='hard')
ensemble.fit(X_train, y_train)

from sklearn.model_selection import cross_val_predict
predictions = cross_val_predict(ensemble, X, y)

import scikitplot as skplt
skplt.metrics.plot_confusion_matrix(y, predictions, normalize=True)
plt.show()

from Stacking import EnsembleClassifier
np.random.seed(123)
eclf = EnsembleClassifier(clfs=[clf4,gb_clf,clf2], weights=[1,1,1])

from sklearn.model_selection import cross_val_predict
predictions = cross_val_predict(eclf, X, y)

import scikitplot as skplt
skplt.metrics.plot_confusion_matrix(y, predictions, normalize=True)
plt.show()