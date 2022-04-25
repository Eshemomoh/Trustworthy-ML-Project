# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2  18:16:47 2022

@author: Lucky Yerimah
"""
import numpy as np
import scipy as sp
from sklearn import linear_model,metrics
import sklearn.metrics.pairwise
import tensorflow as tf
from tqdm import trange
from tensorflow.keras.optimizers import Adam

#tf.random.set_seed(1)
#np.random.seed(1)
#%%
# Build linear model as explainer
def buildModel(Nfeatures):
    xin = tf.keras.Input(shape=(Nfeatures,))
    x = tf.keras.layers.Dense(1,use_bias=False,dtype=tf.float64)(xin)
    return tf.keras.Model(inputs=xin,outputs=x)

#%%
# Return words and their values
def get_words(vectorizer,keys,lime=False):
    
    Words = []
    value = []
    if lime:
        for a,b in keys:
            Words.append(str(a))
            value.append(b)
        
    else:
        feature_words = vectorizer.get_feature_names_out()
        for a,b in keys:
            word = feature_words[a]
            Words.append(word)
            value.append(b)
        
    return Words,value
#%%
# Obtained from original LIME paper implementations
# for sampling around x', generating z' and f(z)

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


# This is GPLIME
class GPLIME:
    def __init__(self, num_samples=5000):
        
        # Kernel for defining locality of x'
        rho = 25
        kernel = lambda d: np.sqrt(np.exp(-(d**2)/rho**2))
        
        
        self.kernel_fn = kernel
        self.data_labels_distances_mapping_fn = data_labels_distances_mapping_text
        self.num_samples = num_samples
        self.mse = tf.keras.losses.MeanSquaredError()
        self.opt = Adam(learning_rate=0.01)

    def data_labels_distances_mapping(self, raw_data, classifier_fn):
        data, labels, distances, mapping = self.data_labels_distances_mapping_fn(raw_data, classifier_fn, self.num_samples)
        return data, labels, distances, mapping

# Train GPLIME model
    def train(self,data,labels,gamma,alpha):
        
        #GP Kernel K(x,x)
       
        kernel = tf.linalg.matmul(data,data,transpose_a=True)
        with tf.GradientTape() as tape:
            prediction = self.model(data)
            # loss = least square loss + L1 norm + GP term
            loss = self.mse(labels,prediction) + gamma*tf.norm(self.model.trainable_weights,ord=1) \
                - alpha*(tf.math.log(tf.linalg.det(kernel))\
                -0.5*(tf.linalg.matmul(tf.linalg.matmul(self.model.trainable_weights,\
                        tf.linalg.pinv(kernel),transpose_a=True),self.model.trainable_weights)))
                    
        gradients = tape.gradient(loss,self.model.trainable_variables)
        self.opt.apply_gradients(zip(gradients,self.model.trainable_variables))
        return loss
    
    def fitmodel(self,data,labels,gamma,alpha,Epoch):
        
        Loss = []
        for i in trange(Epoch):
            loss = self.train(data,labels,gamma,alpha)
            Loss.append(np.mean(loss.numpy()))
        
        return Loss,self.model.trainable_variables
            
        
    
    def explain_instance_with_data(self,data, labels, distances, label, num_features):
        
        weights = self.kernel_fn(distances)
        weighted_data = data * weights[:, np.newaxis]
        weighted_labels = labels * weights        
        
        # Build model using Tensorflow
        self.model = buildModel(weighted_data.shape[1])
        
        # Call fit to train model and obtain weights
        Loss, coefs = self.fitmodel(weighted_data,weighted_labels,0.9,0.1,100)
        
        # Select K highest weights as the most important
        coefs = np.array(coefs)
        coefs = np.squeeze(coefs,axis=0)
        coefs_sq = coefs**2
        sorted_coefs = np.sort(coefs_sq)
        selected_coefs = sorted_coefs[-num_features:]
        inde, rows = np.where(coefs_sq == np.squeeze(selected_coefs))
        used_features = inde
        
        # Use index of selected weights to select features to train a linear model for explanation
        debiased_model = linear_model.Ridge(alpha=0, fit_intercept=False)
        debiased_model.fit(weighted_data[:, used_features], weighted_labels)
        
        
        return sorted(zip(used_features, debiased_model.coef_), key=lambda x:np.abs(x[1]), reverse=True),Loss
    
    # for explaining instances
    def explain_instance(self, raw_data,label, classifier_fn, num_features, dataset=None):
        
        # Obtain data and distances that define specific locality
        data, labels, distances, mapping = self.data_labels_distances_mapping(raw_data, classifier_fn)
        # explain instance and return training loss
        exp,Loss =   self.explain_instance_with_data(data, labels, distances, label, num_features)
        exp = [(mapping[x[0]], x[1]) for x in exp]
        
        return exp,Loss
        
# # # This is a 
# class GPLIME():
    
#     def __init__(self):


#         self.mse = tf.keras.losses.MeanSquaredError()
#         self.opt = Adam(learning_rate=0.001)


# # Train GPLIME model
#     def train(self,data,labels,gamma,alpha):
        
#         #GP Kernel K(x,x)
#         D = tf.Variable(tf.ones([kernel.shape],dtype=tf.float64),trainable=True)
#         kernel = tf.linalg.matmul(data,tf.linalg.matmul(D,data,transpose_a=True),transpose_a=True)
#         with tf.GradientTape() as tape:
#             prediction = self.model(data)
            
#             # loss = least square loss + L1 norm + GP term
#             loss = self.mse(labels,prediction) + gamma*tf.norm(self.model.trainable_weights,ord=1) \
#                 + alpha*(tf.math.log(tf.linalg.det(kernel))\
#                 +0.5*(tf.linalg.matmul(tf.linalg.matmul(self.model.trainable_weights,\
#                         tf.linalg.pinv(kernel),transpose_a=True),self.model.trainable_weights)))
                    
#         gradients = tape.gradient(loss,self.model.trainable_variables)
#         self.opt.apply_gradients(zip(gradients,self.model.trainable_variables))
#         return loss
    
#     def fitmodel(self,data,labels,gamma,alpha,Epoch):
        
#         Loss = []
#         for i in trange(Epoch):
#             loss = self.train(data,labels,gamma,alpha)
#             Loss.append(np.mean(loss.numpy()))
        
#         return Loss,self.model.trainable_variables
            
        
#     # Explain instances
#     def explain_instance(self,classifier,train_data,data, label, num_features):
        
#         train_labels = classifier(train_data)
              
        
#         # Build model using Tensorflow
#         self.model = buildModel(train_data.shape[1])
        
#         # Call fit to train model and obtain weights
#         Loss, coefs = self.fitmodel(train_data,train_labels,0.9,0.01,500)
        
#         # Select K highest weights as the most important
#         coefs = np.array(coefs)
#         coefs = np.squeeze(coefs,axis=0)
#         coefs_sq = coefs**2
#         sorted_coefs = np.sort(coefs_sq)
#         selected_coefs = sorted_coefs[-num_features:]
#         inde, rows = np.where(coefs_sq == np.squeeze(selected_coefs))
#         used_features = inde
        
#         # Use index of selected weights to select features to train a linear model for explanation
#         debiased_model = linear_model.Ridge(alpha=0, fit_intercept=False)
#         debiased_model.fit(data[:, used_features], label)
        
#         exp = sorted(zip(used_features, debiased_model.coef_), key=lambda x:np.abs(x[1]), reverse=True)
#         Explanation = [(mapping[x[0]], x[1]) for x in exp]
#         return Explanation,Loss
    
        
