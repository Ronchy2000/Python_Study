# -*- coding: utf-8 -*-
# @Time    : 2022/7/12 10:03
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : Hello_decisionTree.py
# @Software: PyCharm
'''
决策树具有随机性
random_state  设置分枝中的随机模式的参数
,splitter,

'''
import pandas as pd
import graphviz
from sklearn import tree
from sklearn.datasets import load_wine  #生成数据集
from sklearn.model_selection import train_test_split  #划分训练测试集

'''
建模三步走
clf = tree.DecisionTreeClassifier()
clf = clf.fit(x_train,y_train)  #训练模型
result = clf.score(x_test,y_test)  #导入测试集，得到模型评估指标
'''

wine = load_wine()
print("wine:",wine)
print("wine_data；",wine.data,end='\n') #数据

# print("wine_target:",wine.target,end='\n') #标签

# df = pd.concat([pd.DataFrame(wine.data),pd.DataFrame(wine.target)],axis=1)
# print(df)

print("wine.feature_names:",wine.feature_names) #特征的名字

# print("wine.target_names:",wine.target_names)  #标签的名字  -> ['class_0' 'class_1' 'class_2']

Xtrain , Xtest , Ytrain , Ytest = train_test_split(wine.data,wine.target,test_size = 0.3)  #30% 作为测试集
print("Xtrain.shape:",Xtrain.shape)  #124行
print("wine.data.shape:",wine.data.shape)  #178行
print("Ytrain.shape:",Ytrain.shape)  #124行
print(Ytrain)
print("wine.target.shape:",wine.target.shape)  #178行


#
# clf = tree.DecisionTreeClassifier(criterion="entropy",random_state=30 , splitter="random") #random_state， 随机选取特征
# # clf = tree.DecisionTreeClassifier(criterion="entropy")
# clf = clf.fit(Xtrain,Ytrain)
# score = clf.score(Xtest,Ytest)
# print("score:",score)
#
#
# feature_name = ['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 'total_phenols'
#                ,'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins', 'color_intensity', 'hue', 'od280/od315_of_diluted_wines', 'proline']
# dot_data = tree.export_graphviz(clf
#                                 ,feature_names= feature_name
#                                 ,class_names=["Gin","Sherry","Vermouth"]
#                                 ,filled=True  #颜色
#                                 ,rounded=True
#                                 )
# graph = graphviz.Source(dot_data)
# graph.view()   #查看流程图
#
# # print(clf.feature_importances_)
# print( [*zip(feature_name,clf.feature_importances_)]  )
#
#
#



