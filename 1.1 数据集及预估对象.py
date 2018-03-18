# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 10:07:14 2018

@author: Lenovo
Python 3.6
"""

#iris数据集
from sklearn import datasets
iris=datasets.load_iris()
data=iris.data
print(data.shape)
"""
(150.4)
"""


#digits数据集
digits=datasets.load_digits()
print(digits.images.shape)
"""
(1797, 8, 8)
"""

#将8x8的图像转化成长度为64的特征向量
data=digits.images.reshape((digits.images.shape[0],-1))
print(data.shape)
"""
(1797, 64)
"""
print('------------')


#通过fit的方法进行拟合数据
estimator.fit(data)

#拟合模型对象构造参数
estimator=Estimator(param1=1,param2=2)

#拟合参数


