# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 19:41:32 2022
New implementation of GP-LIME explainer

@author: lucky
"""

import numpy as np
import matplotlib.pyplot as plt
import sklearn.ensemble as ensemble
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from tqdm import trange
from sklearn import tree,metrics
from explainers import GPLIME 
import scipy as sp
#%%
categories = ['alt.atheism','soc.religion.christian']
train_group = fetch_20newsgroups(subset='train',categories=categories)
test_group = fetch_20newsgroups(subset='test',categories=categories)
class_names = ['atheism', 'christian']

# vectorize text
vectorizer = TfidfVectorizer(lowercase=False)
train_vectors = vectorizer.fit_transform(train_group.data)
test_vectors = vectorizer.transform(test_group.data)



#%%
# Obtained from original LIME paper implementations

def data_labels_distances_mapping_text(x, classifier_fn, num_samples):
    distance_fn = lambda x : metrics.pairwise.cosine_distances(x[0],x)[0] * 100
    features = x.nonzero()[1]
    vals = np.array(x[x.nonzero()])[0]
    doc_size = len(sp.sparse.find(x)[2])                                    
    sample = np.random.randint(1, doc_size, num_samples - 1)                             
    data = np.zeros((num_samples, len(features)))    
    inverse_data = np.zeros((num_samples, len(features)))                                         
    data[0] = np.ones(doc_size)
    inverse_data[0] = vals
    features_range = range(len(features)) 
    for i, s in enumerate(sample, start=1):                                               
        active = np.random.choice(features_range, s, replace=False)                       
        data[i, active] = 1
        for j in active:
            inverse_data[i, j] = 1
    sparse_inverse = sp.sparse.lil_matrix((inverse_data.shape[0], x.shape[1]))
    sparse_inverse[:, features] = inverse_data
    sparse_inverse = sp.sparse.csr_matrix(sparse_inverse)
    mapping = features
    labels = classifier_fn(sparse_inverse)
    distances = distance_fn(sparse_inverse)
    return data, labels, distances, mapping
#%%
# # Train a random forest classifier as the original classifier
# rf = ensemble.RandomForestClassifier(n_estimators=500)
# rf.fit(train_vectors, train_group.target)

#%%
# Train a decision tree model 
dtree = tree.DecisionTreeClassifier()
dtree.fit(train_vectors, train_group.target)
pred = dtree.predict(test_vectors)
print(metrics.f1_score(test_group.target, pred, average='binary'))


#%% 
## IMPLEMENT LIME ALGORITHM 

#  
explainer = GPLIME(kernel,train_vectors.shape[1],data_labels_distances_mapping_text)

index = np.arange(1)
explaininstance,loss = explainer.explain_instance(test_vectors[index,:],test_group.target[index],dtree.predict,10)
#%%
plt.figure()
plt.plot(loss)
plt.show()
#%%

def returnnames(vectorizer,keys):
    featurenames = vectorizer.get_feature_names_out()
    Words = []
    value = []
    for a,b in keys:
        word = names[a]
        Words.append(word)
        value.append(b)
        
    return Words,value
#data, labels, distances, mapping =  data_labels_distances_mapping_text(test_vectors[index,:], dtree.predict,10)









