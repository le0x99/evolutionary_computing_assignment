#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 11:33:53 2020

@author: leonarddariusvorbeck
"""

## First Simulation (Using MLP controller)
import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, 'evoman')
from custom_players import player_1
from evoman.environment import Environment
experiment_name = 'MLP_SIM_0'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# Initialize random population
npop = 100
n_hidden = 4
pop = [ {"bias_hidden" : np.random.randn(1, n_hidden),
         "bias_output" : np.random.randn(1, 5),
         "weights_1" : np.random.randn(20, n_hidden),
         "weights_2" : np.random.randn(n_hidden, 5),
         "ID" : _ } for _ in range(npop)]



# initializes environment 

def run(genes, en=3):
    env = Environment(experiment_name="first_try",
    				  playermode="ai",
    				  player_controller=player_1(),
    			  	  speed="fastest",
    				  enemymode="static",
    				  level=2,
    				  enemies=[en])
    return env.play(genes)

