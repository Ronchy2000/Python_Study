# -*- coding: utf-8 -*-
# @Time    : 2022/7/13 17:10
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : 随机森林.py
# @Software: PyCharm
'''
以决策树为基学习器构建Bagging集成的基础上，进一步在决策树训练的过程中引入了随机属性选择

集成学习: ensemble learning  集成算法
'''

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split


wine = load_wine()

print(wine.data[0])
print(wine.target)
print("wine:",type(wine.data))

Xtrain,Xtest,Ytrain,Ytest = train_test_split(wine.data,wine.target,test_size=0.3)  #测试集 30%

clf = DecisionTreeClassifier(random_state = 0)
rfc = RandomForestClassifier(random_state = 0)

clf = clf.fit(Xtrain,Ytrain)
rfc = rfc.fit(Xtrain,Ytrain)

score_c = clf.score(Xtest,Ytest)
score_r = rfc.score(Xtest,Ytest)

print("single Tree:{}\n".format(score_c)
    ,"Random Forest:{}".format(score_r)
      )


from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

rfc_l = []
clf_l = []
for i in range(10):
    rfc = RandomForestClassifier(n_estimators=25)
    rfc_s = cross_val_score(rfc,wine.data,wine.target,cv=10).mean()
    rfc_l.append(rfc_s)

    clf = DecisionTreeClassifier()
    clf_s = cross_val_score(clf,wine.data,wine.target,cv=10).mean()
    clf_l.append(clf_s)

plt.plot(range(1,11),rfc_l,label="RandomForest")
plt.plot(range(1,11),clf_l,label="Decision Tree")
plt.legend()
plt.show()


#n_estimators 的学习曲线
#----------------------------------------------------------------------
# superpa =[]
# for i in range(20):
#     rfc = RandomForestClassifier(n_estimators=i+1,n_jobs=-1)
#     rfc_s = cross_val_score(rfc,wine.data,wine.target,cv=10).mean()
#     superpa.append(rfc_s)
# print(max(superpa),superpa.index(max(superpa)))
# plt.figure(figsize=[20,5])
# plt.plot(range(1,21),superpa)
# plt.show()