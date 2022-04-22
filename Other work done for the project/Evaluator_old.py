# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 17:12:35 2022

@author: lucky
"""
import sys
import copy
sys.path.append('..')
import argparse
import parzen_windows
import numpy as np
import pickle
import sklearn
from sklearn.metrics import accuracy_score

#%%



class ExplanationEvaluator:
  def __init__(self, classifier_names=None):
    self.classifier_names = classifier_names
    if not self.classifier_names:
      self.classifier_names = ['l1lime', 'tree']
    self.classifiers = {}
  def init_classifiers(self, dataset):
    self.classifiers[dataset] = {}
    for classifier in self.classifier_names:
      if classifier == 'l1lime':
        try_cs = np.arange(.1,0,-.01)
        for c in try_cs:
          self.classifiers[dataset]['l1logreg'] = linear_model.LogisticRegression(penalty='l1', fit_intercept=True, C=c)
          self.classifiers[dataset]['l1logreg'].fit(self.train_vectors[dataset], self.train_labels[dataset])
          lengths = [len(x.nonzero()[0]) for x in self.classifiers[dataset]['l1logreg'].transform(self.train_vectors[dataset])]
          if np.max(lengths) <= 10:
            #print 'Logreg for ', dataset, ' has mean length',  np.mean(lengths), 'with C=', c
            #print 'And max length = ', np.max(lengths)
            break
      if classifier == 'tree':
        self.classifiers[dataset]['tree'] = tree.DecisionTreeClassifier(random_state=1)
        self.classifiers[dataset]['tree'].fit(self.train_vectors[dataset], self.train_labels[dataset])
        lengths = [len(get_tree_explanation(self.classifiers[dataset]['tree'], self.train_vectors[dataset][i])) for i in range(self.train_vectors[dataset].shape[0])]
        #print 'Tree for ', dataset, ' has mean length',  np.mean(lengths)
  def load_datasets(self, dataset_names):
    self.train_data = {}
    self.train_labels = {}
    self.test_data = {}
    self.test_labels = {}
    for dataset in dataset_names:
      self.train_data[dataset], self.train_labels[dataset], self.test_data[dataset], self.test_labels[dataset], _ = LoadDataset(dataset)
  def vectorize_and_train(self):
    self.vectorizer = {}
    self.train_vectors = {}
    self.test_vectors = {}
    self.inverse_vocabulary = {}
    print 'Vectorizing...', 
    for d in self.train_data:
      self.vectorizer[d] = CountVectorizer(lowercase=False, binary=True)
      self.train_vectors[d] = self.vectorizer[d].fit_transform(self.train_data[d])
      self.test_vectors[d] = self.vectorizer[d].transform(self.test_data[d])
      terms = np.array(list(self.vectorizer[d].vocabulary_.keys()))
      indices = np.array(list(self.vectorizer[d].vocabulary_.values()))
      self.inverse_vocabulary[d] = terms[np.argsort(indices)]
    print 'Done'
    print 'Training...'
    for d in self.train_data:
      print d
      self.init_classifiers(d)
    print 'Done'
    print
  def measure_explanation_hability(self, explain_fn, max_examples=None):
    """Asks for explanations for all predictions in the train and test set, with
    budget = size of explanation. Returns two maps (train_results,
    test_results), from dataset to classifier to list of recalls"""
    budget = 10
    train_results = {}
    test_results = {}
    for d in self.train_data:
      train_results[d] = {}
      test_results[d] = {}
      print 'Dataset:', d
      for c in self.classifiers[d]:
        train_results[d][c] = []
        test_results[d][c] = []
        if c == 'l1logreg':
          c_features = self.classifiers[d][c].coef_.nonzero()[1]
        print 'classifier:', c 
        for i in range(len(self.test_data[d])):
          if c == 'l1logreg':
            true_features = set([x for x in self.test_vectors[d][i].nonzero()[1] if x in c_features])
          elif c == 'tree':
            true_features = get_tree_explanation(self.classifiers[d][c], self.test_vectors[d][i])
          if len(true_features) == 0:
            continue
          to_get = budget
          exp_features = set(map(lambda x:x[0],
          explain_fn(self.test_vectors[d][i], self.test_labels[d][i] ,self.classifiers[d][c], to_get, d)))
          test_results[d][c].append(float(len(true_features.intersection(exp_features))) / len(true_features))
          if max_examples and i >= max_examples:
            break
    return train_results, test_results
