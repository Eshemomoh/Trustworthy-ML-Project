# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 22:45:25 2022

@author: lucky
"""
import numpy as np
from explainers import LIME, GPLIME
import scipy as sp
#%%



#%%
# Parameters for defining locality 
rho = 25
kernel = lambda d: np.sqrt(np.exp(-(d**2)/rho**2))


#%%

# Implement GPLIME
explainer 

