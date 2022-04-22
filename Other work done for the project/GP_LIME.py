# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 12:36:16 2022

@author: lucky
"""

import numpy as np
import matplotlib.pyplot as plt
import sklearn.ensemble as ensemble
#from sklearn.datasets import load_boston
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from tqdm import trange
from sklearn import tree



#%%
#load dataset
# boston = load_boston()
# train, test, labels_train, labels_test = train_test_split(boston.data, boston.target, train_size=0.80)
housing = fetch_california_housing()
train, test, labels_train, labels_test = train_test_split(housing.data, housing.target, train_size=0.20)
# use a random forest as the original model
scaler = StandardScaler()
scaler.fit(train)
train = scaler.transform(train)
test = scaler.transform(test)
rf = ensemble.RandomForestRegressor(n_estimators=1000)

#%%
# Train Random Forest
# rf.fit(train, labels_train)
# print("Random Forest MSE = ", np.mean((rf.predict(test)-labels_test)**2))

# #%%
# from sklearn.datasets import load_boston
# boston = load_boston()
# categorical_features = np.argwhere(np.array([len(set(boston.data[:,x])) for x in range(boston.data.shape[1])]) <= 10).flatten()


#%%
# Define LIME as a Tensorflow model 
def buildModel(Nfeatures):
    xin = tf.keras.Input(shape=(Nfeatures,))
    x = Dense(1,use_bias=False,dtype=tf.float64)(xin)
    return tf.keras.Model(inputs=xin,outputs=x)
    

class GPLIME():
    def __init__(self,WKernel):
        super(GPLIME,self).__init__()
        self.model = buildModel(8)
        #self.lime = Dense(1,use_bias=False,kernel_regularizer='l1') # no activation, no bias, L1 regularizer 
        self.mse = tf.keras.losses.MeanSquaredError()
        self.gpr = GaussianProcessRegressor(kernel = WKernel)
        self.opt = Adam(learning_rate=0.1)
    
    def train(self,data,labels,gamma):
        
        kernel = tf.linalg.matmul(data,data,transpose_a=True)
        with tf.GradientTape() as tape:
            prediction = self.model(data)
            loss = self.mse(labels,prediction) + gamma*tf.norm(self.model.trainable_weights,ord=1) \
                + gamma*(tf.math.log(tf.linalg.det(kernel))+0.5*(tf.linalg.matmul(tf.linalg.matmul(self.model.trainable_weights,kernel,transpose_a=True),
                                                                           self.model.trainable_weights)))
                #+ tf.math.log(tf.linalg.det(kernel))+0.5*(tf.linalg.matmul(tf.linalg.matmul(self.model.trainable_weights,kernel,transpose_a=True),
                                                                           #self.model.trainable_weights,kernel,transpose_b=True))
        
        gradients = tape.gradient(loss,self.model.trainable_variables)
        self.opt.apply_gradients(zip(gradients,self.model.trainable_variables))
        return loss
        
    def call(self,x):
        y = self.model(x)
        return y
    
    
            
    
#%%
kernel = WhiteKernel()
Gpmodel = GPLIME(kernel)

#%%
Epoch = 10
Loss = []
for i in trange(Epoch):
    loss = Gpmodel.train(train,labels_train,0.1)
    print(loss.numpy())
    Loss.append(loss.numpy())
    
#%%    # if i > 2 and (loss-Loss[-1])**2 < 0.01:
    #     break
Weights = Gpmodel.model.weights
print(Weights)
Weights = np.array(Weights)
Weights = np.squeeze(Weights,0)

# index = np.arange(5)
# np.random.shuffle(index)
# instances = test[index]
# instance_label = labels_test[index]

# print('explain1 = ', np.multiply(instances[0],Weights))
# print('explain2 = ', np.multiply(instances[1],Weights))
# print('explain3 = ', np.multiply(instances[2],Weights))
# print('explain4 = ', np.multiply(instances[3],Weights))
# print('explain5 = ', np.multiply(instances[4],Weights))

#%%

#%%

clf = tree.DecisionTreeRegressor()
clf = clf.fit(train,labels_train)

#%%
#print(clf.feature_importances_)
#print(Weights)

names = ['income','age','room#','bed#','Popu','member#','Lat','Long']

plt.figure()
plt.bar(names,clf.feature_importances_)
plt.show()

plt.figure()
plt.bar(names,np.squeeze(Weights,1))
plt.show()