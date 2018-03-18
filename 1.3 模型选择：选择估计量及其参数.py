# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 16:15:15 2018

@author: Lenovo
Python 3.6
"""

#根据score判断拟合度
from sklearn import svm,datasets
digits=datasets.load_digits()
X_digits=digits.data
Y_digits=digits.target
svc=svm.SVC(C=1,kernel='linear')
print(svc.fit(X_digits[:-100],Y_digits[:-100]).score(X_digits[-100:],Y_digits[-100:]))
"""
0.98
"""

print('--------------')

#KFold 交叉验证.
import numpy as np
X_folds = np.array_split(X_digits, 3)
y_folds = np.array_split(Y_digits, 3)
#转化为一个列表的形式
scores = list()
for k in range(3):
    # 为了稍后的 ‘弹出’ 操作，我们使用 ‘列表’ 来复制数据
    X_train = list(X_folds)
    #弹出元素
    X_test  = X_train.pop(k)
    #连接元素
    X_train = np.concatenate(X_train)
    y_train = list(y_folds)
    y_test  = y_train.pop(k)
    y_train = np.concatenate(y_train)
    scores.append(svc.fit(X_train, y_train).score(X_test, y_test))
print(scores)
"""
[0.93489148580968284, 0.95659432387312182, 0.93989983305509184]
"""


#split 进行交叉验证
from sklearn.model_selection import KFold,cross_val_score

X=["a","a","b","c","c","c"]
k_fold=KFold(n_splits=3)
print(k_fold)
"""
KFold(n_splits=3, random_state=None, shuffle=False)
"""
for train_indices, test_indices in k_fold.split(X):
     print('Train: %s | test: %s' % (train_indices, test_indices))
"""
Train: [2 3 4 5] | test: [0 1]
Train: [0 1 4 5] | test: [2 3]
Train: [0 1 2 3] | test: [4 5]
"""


#网格搜索
from sklearn.model_selection import GridSearchCV, cross_val_score
Cs = np.logspace(-6, -1, 10)
clf = GridSearchCV(estimator=svc, param_grid=dict(C=Cs),n_jobs=-1)

print(clf.fit(X_digits[:1000], Y_digits[:1000]))        

print(clf.best_score_)                                  

print(clf.best_estimator_.C)  



#交叉验证估计量
from sklearn import linear_model,datasets
lasso=linear_model.LassoCV()
diabetes=datasets.load_diabetes()
X_diabetes=diabetes.data
y_diabetes=diabetes.target
print(lasso.fit(X_diabetes,y_diabetes))
"""
LassoCV(alphas=None, copy_X=True, cv=None, eps=0.001, fit_intercept=True,
    max_iter=1000, n_alphas=100, n_jobs=1, normalize=False, positive=False,
    precompute='auto', random_state=None, selection='cyclic', tol=0.0001,
    verbose=False)
"""

#估计器自动选择的lambda
print(lasso.alpha_)
"""
0.0122918950875
"""







