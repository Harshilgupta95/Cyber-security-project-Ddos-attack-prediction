import pandas as pd
import matplotlib.pyplot as plt

print('1.UDP\n2.TCPSYN\n3.ICMP\n4.LAND\n5.TCPSYNACK')
attack=input('Attack name: ')
path='C:\\Users\\LENOVO\\Downloads\\'+attack+'.csv'
io = pd.read_csv(path,sep=",",usecols=(9,11,12,13,16,21))
io.head()
import seaborn as sns
sns.countplot(io['Class'],label="Count")
plt.show()

import pylab as pl
io.drop('Class' ,axis=1).hist(bins=30, figsize=(9,9))
pl.suptitle("Histogram for each numeric input variable")
plt.savefig('hist')
plt.show()