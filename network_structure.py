#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 11:29:32 2020

@author: leonarddariusvorbeck
"""

import numpy as np

def sigmoid_activation(x):
	return 1./(1.+np.exp(-x))
def relu_activation(x):
    return np.maximum(0,x)
def softmax_activation(x):
    expo = np.exp(x)
    expo_sum = np.sum(expo)
    return expo/expo_sum

# Random set of weights (the genes)
    
genes = np.random.randn(105)

## NN 0 : 20 input -> 5 output (+5x1 bias per output)
## Params : 20x5 weights + 5x1 bias = 105

# 20 Input Nodes
inputs = np.random.randn(20)
# MinMax Normalization
inputs = (inputs-min(inputs))/float((max(inputs)-min(inputs)))
# 5x1 Biases, one for each output Node
bias = genes[:5].reshape(1, 5)
# 5x20 weights, given 20 inputs and 5 outputs
weights = genes[5:].reshape((len(inputs), 5))
# Output de-linearized by activation
output = sigmoid_activation(inputs.dot(weights) + bias)[0]


## NN 1 : 20 input -> 10 hidden  -> 5 output (+5x1 bias per output + 10x1 bias per hidden)
# 20 Input Nodes
inputs = np.random.randn(20)
# MinMax Normalization
inputs = (inputs-min(inputs))/float((max(inputs)-min(inputs)))