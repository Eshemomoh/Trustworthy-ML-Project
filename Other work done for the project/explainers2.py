# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 18:16:47 2022

@author: lucky
"""
import numpy as np
import scipy as sp
from sklearn import linear_model
import sklearn.metrics.pairwise
import tensorflow as tf
from tqdm import trange
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
#%%
# Build model
def buildModel(Nfeatures):
    xin = tf.keras.Input(shape=(Nfeatures,))
    x = Dense(1,use_bias=False,dtype=tf.float64)(xin)
    return tf.keras.Model(inputs=xin,outputs=x)

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
    def __init__(self,
                 kernel_fn,
                 Nfeatures,
                 data_labels_distances_mapping_fn,
                 num_samples=5000,
                 mean=None,
                 return_mean=True,
                 return_mapped=False,
                 positive=False):
        # Transform_classifier, transform_explainer,
        # transform_explainer_to_classifier all take raw data in, whatever that is.
        # perturb(x, num_samples) returns data (perturbed data in f'(x) form),
        # inverse_data (perturbed data in x form) and mapping, where mapping is such
        # that mapping[i] = j, where j is an index for x form.
        # distance_fn takes raw data in. what we're calling raw data is just x
        self.kernel_fn = kernel_fn
        self.data_labels_distances_mapping_fn = data_labels_distances_mapping_fn
        self.num_samples = num_samples
        self.mean = mean
        self.return_mapped=return_mapped
        self.return_mean = return_mean
        self.positive=positive;
        self.mse = tf.keras.losses.MeanSquaredError()
        self.opt = Adam(learning_rate=0.01)
    def reset(self):
        pass

    def data_labels_distances_mapping(self, raw_data, classifier_fn):
        data, labels, distances, mapping = self.data_labels_distances_mapping_fn(raw_data, classifier_fn, self.num_samples)
        return data, labels, distances, mapping
    def generate_lars_path(self, weighted_data, weighted_labels):
        X = weighted_data
        alphas, active, coefs = linear_model.lars_path(X, weighted_labels, method='lasso', verbose=False, positive=self.positive)
        return alphas, coefs

    def train(self,data,labels,gamma,alpha):
        
        kernel = tf.linalg.matmul(data,data,transpose_a=True)
        with tf.GradientTape() as tape:
            prediction = self.model(data)
            loss = self.mse(labels,prediction) + gamma*tf.norm(self.model.trainable_weights,ord=1) \
                + alpha*(tf.math.log(tf.linalg.det(kernel))+0.5*(tf.linalg.matmul(tf.linalg.matmul(self.model.trainable_weights,kernel,transpose_a=True),
                                                                           self.model.trainable_weights)))
        gradients = tape.gradient(loss,self.model.trainable_variables)
        self.opt.apply_gradients(zip(gradients,self.model.trainable_variables))
        return loss
    
    def fitmodel(self,data,labels,gamma,alpha,Epoch):
        
        Loss = []
        for i in trange(Epoch):
            loss = self.train(data,labels,gamma,alpha)
            Loss.append(np.mean(loss))
        
        return Loss,self.model.trainable_variables
            
        
    
    def explain_instance_with_data(self,data, labels, distances, label, num_features):
        weights = self.kernel_fn(distances)
        weighted_data = data * weights[:, np.newaxis]
        # if self.mean is None:
        #   mean = np.mean(labels[:, label])
        # else:
        #   mean = self.mean
        # shifted_labels = labels[:, label] - mean
        
        weighted_labels = labels * weights
        used_features = range(weighted_data.shape[1])
        nonzero = used_features
        alpha = 1
        
        
        # alphas, coefs = self.generate_lars_path(weighted_data, weighted_labels)
        # for i in range(len(coefs.T) - 1, 0, -1):
        #     nonzero = coefs.T[i].nonzero()[0]
        #     if len(nonzero) <= num_features:
        #         chosen_coefs = coefs.T[i]
        #         alpha = alphas[i]
        #         break
        #used_features = nonzero
        self.model = buildModel(weighted_data.shape[1])
        Loss, coefs = self.fitmodel(weighted_data,weighted_labels,0.5,0.1,100)
        coefs = np.array(coefs)
        coefs = np.squeeze(coefs,axis=0)
        coefs_sq = coefs**2
        sorted_coefs = np.sort(coefs_sq)
        selected_coefs = sorted_coefs[-num_features:]
        inde, rows = np.where(coefs_sq == np.squeeze(selected_coefs))
        used_features = inde
        debiased_model = linear_model.Ridge(alpha=0, fit_intercept=False)
        debiased_model.fit(weighted_data[:, used_features], weighted_labels)
        
        if self.return_mean:
            return sorted(zip(used_features,
                      debiased_model.coef_),
                      key=lambda x:np.abs(x[1]), reverse=True),Loss
        else:
            return sorted(zip(used_features,
                      debiased_model.coef_),
                      key=lambda x:np.abs(x[1]), reverse=True),Loss

    def explain_instance(self,
                         raw_data,
                         label,
                         classifier_fn,
                         num_features, dataset=None):
        if not hasattr(classifier_fn, '__call__'):
            classifier_fn = classifier_fn.predict_proba
        data, labels, distances, mapping = self.data_labels_distances_mapping(raw_data, classifier_fn)
        if self.return_mapped:
            if self.return_mean:
                exp,Loss =   self.explain_instance_with_data(data, labels, distances, label, num_features)
        else:
            exp,Loss =   self.explain_instance_with_data(data, labels, distances, label, num_features)
        exp = [(mapping[x[0]], x[1]) for x in exp]
        if self.return_mean:
            return exp,Loss
        else:
            return exp,Loss
        return self.explain_instance_with_data(data, labels, distances, label, num_features),Loss

