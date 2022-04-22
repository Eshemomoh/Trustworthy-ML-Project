# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 18:36:58 2022
Main file to run Trust ML project experiments

@author: Lucky Yerimah
"""
import numpy as np
import matplotlib.pyplot as plt
import sklearn.ensemble as ensemble
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from explainers import GPLIME, GGPLIME, get_words
from sklearn import tree
from lime.lime_text import LimeTextExplainer
from sklearn.pipeline import make_pipeline

#%%

def experiments():
    
    
    # Load 20 news groups dataset for christian and atheism categories
    
    categories = ['alt.atheism','soc.religion.christian']
    train_group = fetch_20newsgroups(subset='train',categories=categories)
    test_group = fetch_20newsgroups(subset='test',categories=categories)
    class_names = ['atheism', 'christian']
    
    # vectorize text
    vectorizer = TfidfVectorizer(lowercase=False)
    train_vectors = vectorizer.fit_transform(train_group.data)
    test_vectors = vectorizer.transform(test_group.data)


    # Train a decision tree model as the original classifier 
    dtree = tree.DecisionTreeClassifier()
    dtree.fit(train_vectors, train_group.target)
    
    # call explainer
    explainer = GPLIME()
    
    # randomly pick an instance to explain
    instance = np.random.choice(10)
    
    # Number of words to select. I.e. Based on human patience
    K = 15
    
    # Generate explanation for the instance and training loss using GPLIME
    explanation, Train_loss = explainer.explain_instance(test_vectors[instance,:],test_group.target[instance],dtree.predict,K)
    
    #plot Training loss
    plt.figure()
    plt.plot(Train_loss)
    plt.show()
    
    # obtain words from explanation
    Words,value = get_words(vectorizer,explanation)
    
    # make bar plots of values and words
    plt.figure()
    plt.barh(Words,value)
    plt.show()

    
    # Generate explanation using LIME
    
    c = make_pipeline(vectorizer, dtree)
    lime_explain = LimeTextExplainer(class_names=class_names)
    exp = lime_explain.explain_instance(test_group.data[instance],c.predict_proba, num_features=15)
    exp = exp.as_list()
    
    Words,value = get_words(vectorizer,explanation,lime=True)
    
    # make bar plots of values and words
    plt.figure()
    plt.barh(Words,value)
    plt.show()
    
    
    
    # Explanation using GGPLIME
    
    explainer2 = GGPLIME()
    explanation, Train_loss = explainer2.explain_instance(dtree.predict,test_vectors[-100:],test_vectors[instance,:],test_group.target[instance],K)
    
    #plot Training loss
    plt.figure()
    plt.plot(Train_loss)
    plt.show()
    
    # obtain words from explanation
    Words,value = get_words(vectorizer,explanation)
    
    
    #Use original classify to predict the class of the instance
    print("Class = ",dtree.predict(test_vectors[instance,:]))
    print(instance)
    
if __name__ == "__main__":
    experiments()
    # instance 200,418,11 are good