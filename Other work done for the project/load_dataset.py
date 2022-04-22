# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 16:04:08 2022
Trust ML Class project

@author: Lucky Yerimah
"""
mmm
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical



def preprocessing(df):
    # Here we separate and encode target variables as 0,1
    y = df.pop('RiskPerformance')
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)
    # Remove rows where all data is missing
    y = y[(df > -9).any(axis=1)]
    df = df.loc[(df > -9).any(axis=1)]
    # Replace remaining special values with NaN
    df = df[df >= 0]
    df['RiskPerformance'] = y
    return df

# FICO HELOC dataset
class Helo_Dataset():
    
    def __init__(self,filepath='heloc_dataset.csv',processing=preprocessing):
        
        self.processing = processing
        self.filepath = filepath
        #  self.filepath = 'heloc_dataset.csv'
        
        
    def data(self):
        
        self.dataf = pd.read_csv(self.filepath)
        self.data = self.processing(self.dataf.copy())
        return self.data
    
    def dataframe(self):
        # First pop and then add 'Riskperformance' column
        dfcopy = self.dataf.copy()
        col = dfcopy.pop('RiskPerformance')
        dfcopy['RiskPerformance'] = col
        return(dfcopy)
        
    def split(self, random_state=0):
        (data_train, data_test) = train_test_split(self.data, stratify=self.data[:,-1], random_state=random_state)

        x_train = data_train[:,0:-1]
        x_test = data_test[:, 0:-1]
        y_train = data_train[:, -1]
        y_test = data_test[:, -1]

        y_train_b = to_categorical(y_train)
        y_test_b = to_categorical(y_test)

        return (self.data, x_train, x_test, y_train_b, y_test_b)
    