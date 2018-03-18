# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 08:16:32 2018

@author: Lenovo
"""

from sklearn import datasets
iris=datasets.load_iris()
digits=datasets.load_digits()

print(digits.data)
print('-------------')
"""
[[  0.   0.   5. ...,   0.   0.   0.]
 [  0.   0.   0. ...,  10.   0.   0.]
 [  0.   0.   0. ...,  16.   9.   0.]
 ..., 
 [  0.   0.   1. ...,   6.   0.   0.]
 [  0.   0.   2. ...,  12.   0.   0.]
 [  0.   0.  10. ...,  12.   1.   0.]]
"""


#每个数字的真实类别,array类型
print(digits.target)
print('--------------')
"""
[0 1 2 ..., 8 9 8]
"""



#二维数组的情况
print(digits.images[0])
print('-------------')
"""
[[  0.   0.   5.  13.   9.   1.   0.   0.]
 [  0.   0.  13.  15.  10.  15.   5.   0.]
 [  0.   3.  15.   2.   0.  11.   8.   0.]
 [  0.   4.  12.   0.   0.   8.   8.   0.]
 [  0.   5.   8.   0.   0.   9.   8.   0.]
 [  0.   4.  11.   0.   1.  12.   7.   0.]
 [  0.   2.  14.   5.  10.  12.   0.   0.]
 [  0.   0.   6.  13.  10.   0.   0.   0.]]
"""



#利用估计器SVC进行分类
from sklearn import svm
clf=svm.SVC(gamma=0.001,C=100)
#产生一个除最后条目的一个训练集进行训练
clf.fit(digits.data[:-1],digits.target[:-1])
print('-------------')
"""
SVC(C=100, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma=0.001, kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
"""

print(clf.predict(digits.data[-1:]))
print('-------------')
"""
[8]
"""



from sklearn import svm
from sklearn import datasets

clf=svm.SVC()
iris=datasets.load_iris()
X,y=iris.data,iris.target
clf.fit(X,y)
print(clf)
"""
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
"""

#将模型进行持久化保存
import pickle
s=pickle.dumps(clf)

clf2=pickle.loads(s)
print(clf2.predict(iris.data[0:5]))
print(iris.target[0:5])
print('---------------')
"""
[0 0 0 0 0]
[0 0 0 0 0]
"""






import numpy as np
from sklearn.svm import SVC

rng=np.random.RandomState(0)
X=rng.rand(100,10)
y=rng.binomial(1,0.5,100)
X_test=rng.rand(5,10)


clf=SVC()
print(clf.set_params(kernel='linear').fit(X,y))
print('-------------')
"""
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
"""

print(clf.predict(X_test))
"""
[1 0 1 1 0]
"""

print(clf.set_params(kernel='rbf').fit(X,y))
"""
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
"""

print(clf.predict(X_test))
print('--------------')
"""
[0 0 0 1 0]
"""


#多分类与多标签拟合
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelBinarizer

X = [[1, 2], [2, 4], [4, 5], [3, 2], [3, 1]]
y = [0, 0, 1, 1, 2]

#变成一维变量
classif = OneVsRestClassifier(estimator=SVC(random_state=0))
print(classif.fit(X, y).predict(X))
"""
[0 0 1 1 2]
"""


#标签二值化后变成二维变量
y = LabelBinarizer().fit_transform(y)
print(classif.fit(X, y).predict(X))
"""
[[1 0 0]
 [1 0 0]
 [0 1 0]
 [0 0 0]
 [0 0 0]]
"""



#变成多个标签
from sklearn.preprocessing import MultiLabelBinarizer
y = [[0, 1], [0, 2], [1, 3], [0, 2, 3], [2, 4]]
y = MultiLabelBinarizer().fit_transform(y)
print(classif.fit(X, y).predict(X))
"""
[[1 1 0 0 0]
 [1 0 1 0 0]
 [0 1 0 1 0]
 [1 0 1 0 0]
 [1 0 1 0 0]]
"""



















