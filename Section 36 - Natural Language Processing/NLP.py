# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 18:02:06 2020

@author: Admin
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter= '\t', quoting= 3)
corpus=[]
for i in range(0,1500):
    review= re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review= review.lower()
    review= review.split()
    ps=PorterStemmer()
    review= [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review=' '.join(review)
    corpus.append(review)
    
cv= CountVectorizer(max_features= 1500)
x=cv.fit_transform(corpus).toarray()
y=dataset.iloc[:, 1].values

from sklearn.model_selection import train_test_split
xtrain, xtest,ytrain,ytest= train_test_split(x,y, test_size=0.2)

from sklearn.tree import DecisionTreeClassifier
dc= DecisionTreeClassifier(criterion='entropy', random_state=0)
dc.fit(xtrain,ytrain)
ypred= dc.predict(xtest)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(ytest,ypred)
