#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 11:29:32 2020

@author: leonarddariusvorbeck
"""

import numpy as np
from evoman.controller import Controller

def sigmoid_activation(x):
	return 1./(1.+np.exp(-x))
def relu_activation(x):
    return np.maximum(0,x)
def softmax_activation(x):
    expo = np.exp(x)
    expo_sum = np.sum(expo)
    return expo/expo_sum

# Random set of weights (the genes)  
genes = {
         "bias" : np.random.randn(1, 5),
         "weights" : np.random.randn(20, 5),
        }

class player_0(Controller):
    def __init__(self, normalize):
        self.normalize = normalize
        return
    def control(self, inputs, genes):
        ## NN 0 : 20 input -> 5 output (+5x1 bias per output)
        ## Params : 20x5 weights + 5x1 bias = 105
        if self.normalize:
            # MinMax Normalization
            inputs = (inputs-min(inputs))/float((max(inputs)-min(inputs)))

        output = sigmoid_activation(inputs.dot(genes["weights"]) + genes["bias"])[0]
        
        # Decisions based on NN output
        if output[0] > 0.5:
            left = 1
        else:
            left = 0
        
        if output[1] > 0.5:
            right = 1
        else:
            right = 0
        
        if output[2] > 0.5:
            jump = 1
        else:
            jump = 0
        
        if output[3] > 0.5:
            shoot = 1
        else:
            shoot = 0
        
        if output[4] > 0.5:
            release = 1
        else:
            release = 0
        
        return [left, right, jump, shoot, release]





class player_1(Controller):
    def __init__(self, normalize):
        self.normalize = normalize
        return
    def control(self, inputs, genes:dict):
        ## NN 1 : 20 input -> 10 hidden -> 5 output 
        ## Params : 20x10x5 weights + 5x1 output bias + 10x1 hidden bias = 1015

        # MinMax Normalization
        if self.normalize:
            inputs = (inputs-min(inputs))/float((max(inputs)-min(inputs)))
        output1 = sigmoid_activation(inputs.dot(genes["weights_1"]) + genes["bias_hidden"])
        output = sigmoid_activation(output1.dot(genes["weights_2"])+ genes["bias_output"])[0]
        
        # Decisions based on NN output
        if output[0] > 0.5:
            left = 1
        else:
            left = 0
        
        if output[1] > 0.5:
            right = 1
        else:
            right = 0
        
        if output[2] > 0.5:
            jump = 1
        else:
            jump = 0
        
        if output[3] > 0.5:
            shoot = 1
        else:
            shoot = 0
        
        if output[4] > 0.5:
            release = 1
        else:
            release = 0
        
        return [left, right, jump, shoot, release]