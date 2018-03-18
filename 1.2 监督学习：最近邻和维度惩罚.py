# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 10:48:02 2018

@author: Lenovo
Python 3.6
"""

import numpy as np
from sklearn import datasets

iris=datasets.load_iris()
iris_x=iris.data
iris_y=iris.target

#显示所以y 的分类
print(np.unique(iris_y))
"""
[0 1 2]
"""

#将iris分为测试集和训练集
np.random.seed(0)
#随机排列，用于使分解的数据随机分布
indices=np.random.permutation(len(iris_x))

#训练数据
iris_x_train=iris_x[indices[:-10]]
iris_y_train=iris_y[indices[:-10]]

#测试数据
iris_x_test=iris_x[indices[-10:]]
iris_y_test=iris_y[indices[-10:]]


#创建和拟合一个knn
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier()
print(knn.fit(iris_x_train,iris_y_train))
"""
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_jobs=1, n_neighbors=5, p=2,
           weights='uniform')
"""

print(knn.predict(iris_x_test))
print(iris_y_test)
"""
[1 2 1 0 0 0 2 1 2 0]
[1 1 1 0 0 0 2 1 2 0]
"""


print('-----------')

#糖尿病数据集
diabetes=datasets.load_diabetes()

diabetes_x_train=diabetes.data[:-20]
diabetes_y_train=diabetes.target[:-20]

diabetes_x_test=diabetes.data[-20:]
diabetes_y_test=diabetes.target[-20:]


#线性回归
from sklearn import linear_model

regr=linear_model.LinearRegression()
print(regr.fit(diabetes_x_train,diabetes_y_train))
"""
LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
"""

#回归系数
print(regr.coef_)
"""
[  3.03499549e-01  -2.37639315e+02   5.10530605e+02   3.27736980e+02
  -8.14131709e+02   4.92814588e+02   1.02848452e+02   1.84606489e+02
   7.43519617e+02   7.60951722e+01]
"""


#均方误差,其中mean表示求平均值
print(np.mean((regr.predict(diabetes_x_test)-diabetes_y_test)**2))
"""
2004.56760269
"""

#方差分数：1表示完美预测，0表示没有任何关系
print(regr.score(diabetes_x_test,diabetes_y_test))
"""
0.585075302269
"""
print('--------')


#收缩


#c_表示按列进行连接
X=np.c_[.5,1].T
y=[.5,1]
test=np.c_[0,2].T
regr=linear_model.LinearRegression()

import matplotlib.pyplot as plt
plt.figure()
np.random.seed(0)
for _ in range(6):
    this_X = .1*np.random.normal(size=(2, 1)) + X
    regr.fit(this_X, y)
    plt.plot(test, regr.predict(test)) 
    plt.scatter(this_X, y, s=3)     
    
print('----------')


regr = linear_model.Ridge(alpha=.1)

plt.figure() 

np.random.seed(0)
for _ in range(6): 
    this_X = .1*np.random.normal(size=(2, 1)) + X
    regr.fit(this_X, y)
    plt.plot(test, regr.predict(test)) 
    plt.scatter(this_X, y, s=3) 



print('------------')




#分类
#尝试用最近邻和线性模型分类数字数据集。留出最后 10%的数据，并测试观察值预期效果


from sklearn import datasets, neighbors, linear_model

digits = datasets.load_digits()
X_digits = digits.data
y_digits = digits.target

n_samples = len(X_digits)

X_train = X_digits[:int(.9 * n_samples)]
y_train = y_digits[:int(.9 * n_samples)]
X_test = X_digits[int(.9 * n_samples):]
y_test = y_digits[int(.9 * n_samples):]

knn = neighbors.KNeighborsClassifier()
logistic = linear_model.LogisticRegression()

print('KNN score: %f' % knn.fit(X_train, y_train).score(X_test, y_test))
print('LogisticRegression score: %f' % logistic.fit(X_train, y_train).score(X_test, y_test))
"""
KNN score: 0.961111
LogisticRegression score: 0.938889
"""






print('-------------')

#线性svms

from sklearn import svm
svc=svm.SVC(kernel='linear')
svc.fit(iris_x_train,iris_y_train)
print(svc)
"""
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
"""


#根据特征1和特征2，尝试用 SVMs 把1和2类从鸢尾属植物数据集中分出来。
#为每一个类留下10%，并测试这些观察值预期效果。

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, svm

iris = datasets.load_iris()
X = iris.data
y = iris.target

X = X[y != 0, :2]
y = y[y != 0]

n_sample = len(X)

np.random.seed(0)
order = np.random.permutation(n_sample)
X = X[order]
y = y[order].astype(np.float)

X_train = X[:int(.9 * n_sample)]
y_train = y[:int(.9 * n_sample)]
X_test = X[int(.9 * n_sample):]
y_test = y[int(.9 * n_sample):]

# fit the model
for fig_num, kernel in enumerate(('linear', 'rbf', 'poly')):
    clf = svm.SVC(kernel=kernel, gamma=10)
    clf.fit(X_train, y_train)

    plt.figure(fig_num)
    plt.clf()
    plt.scatter(X[:, 0], X[:, 1], c=y, zorder=10, cmap=plt.cm.Paired,
                edgecolor='k', s=20)

    # Circle out the test data
    plt.scatter(X_test[:, 0], X_test[:, 1], s=80, facecolors='none',
                zorder=10, edgecolor='k')

    plt.axis('tight')
    x_min = X[:, 0].min()
    x_max = X[:, 0].max()
    y_min = X[:, 1].min()
    y_max = X[:, 1].max()

    XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
    Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(XX.shape)
    plt.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.Paired)
    plt.contour(XX, YY, Z, colors=['k', 'k', 'k'],
                linestyles=['--', '-', '--'], levels=[-.5, 0, .5])

    plt.title(kernel)
plt.show()







