# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 08:30:38 2021
@author: UMAR SADIQUE
"""

import numpy as np
"""Implements a perceptron network"""
class Perceptron(object): # create class for object
    def __init__(self, input_size, lr=1, epochs=100): # initialize input for class (self is for object for which we create class)
        self.W = np.zeros(input_size+1) # input size is 2 but add 1 for bias we put weight value is equal to zero
        self.epochs = epochs # 
        self.lr = lr #
        
    def activation_fn(self, x):
        #return (x >= 0).astype(np.float32)
        return 1 if x >= 0 else 0
    
    def predict(self, x):
        z = self.W.T.dot(x)
        a = self.activation_fn(z)
        return a
    
    def fit(self, X, d):
        for _ in range(self.epochs):
            for i in range(d.shape[0]):
                x = np.insert(X[i], 0, 1)
                y = self.predict(x)
                e = d[i] - y
                self.W = self.W + self.lr * e * x

if __name__ == '__main__':
    X = np.array([[0, 0],[0, 1],[1, 0],[1, 1]])
    d = np.array([0, 0, 0, 1])
    perceptron = Perceptron(input_size=2, lr= 0.5)
    perceptron.fit(X, d)
    print(perceptron.W)