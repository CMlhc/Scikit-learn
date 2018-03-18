# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 17:49:49 2018

@author: Lenovo
"""


#从20个类别中选择4个类别来进行训练
categories = ['alt.atheism', 'soc.religion.christian','comp.graphics', 'sci.med']

from sklearn.datasets import fetch_20newsgroups
twenty_train = fetch_20newsgroups(subset='train',
    categories=categories, shuffle=True, random_state=42)

print(twenty_train.target_names)
"""
['alt.atheism', 'comp.graphics', 'sci.med', 'soc.religion.christian']
"""

print(len(twenty_train.data))
"""
2257
"""

print(len(twenty_train.filenames))
"""
2257
"""


#获取数据的属性
print(twenty_train.target[:10])
"""
[1 1 3 3 3 3 3 2 2 2]
"""

#获取类别名称
for t in twenty_train.target[:10]:
    print(twenty_train.target_names[t])
    
    
"""
comp.graphics
comp.graphics
soc.religion.christian
soc.religion.christian
soc.religion.christian
soc.religion.christian
soc.religion.christian
sci.med
sci.med
sci.med
"""


#使用sckil-learn 来对文本进行分词
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(twenty_train.data)
print(X_train_counts.shape)
"""
(2257, 35788)
"""
print(count_vect.vocabulary_.get(u'algorithm'))
"""
4690
"""

from sklearn.feature_extraction.text import TfidfTransformer
tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)
print(X_train_tf.shape)
"""
(2257, 35788)
"""


"""
更快的方法进行实现
>>> tfidf_transformer = TfidfTransformer()
>>> X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
>>> X_train_tfidf.shape
(2257, 35788)
"""
print('--------------')


#训练多项式分类器
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target)

"""
为了尝试预测新文档所属的类别，我们需要使用和之前同样的步骤来抽取特征。
 不同之处在于，我们在transformer调用 transform 而不是 fit_transform ，
 因为这些特征已经在训练集上进行拟合了:
"""


docs_new = ['God is love', 'OpenGL on the GPU is fast']
X_new_counts = count_vect.transform(docs_new)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)

predicted = clf.predict(X_new_tfidf)

for doc, category in zip(docs_new, predicted):
    print('%r => %s' % (doc, twenty_train.target_names[category]))
    
"""
'God is love' => soc.religion.christian
'OpenGL on the GPU is fast' => comp.graphics
"""
print('----------------')

import numpy as np
twenty_test = fetch_20newsgroups(subset='test',
    categories=categories, shuffle=True, random_state=42)
docs_test = twenty_test.data
predicted = text_clf.predict(docs_test)
print(np.mean(predicted == twenty_test.target))





