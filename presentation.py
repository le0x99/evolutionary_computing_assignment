#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 21:43:57 2020

@author: leonarddariusvorbeck
"""


## First Simulation (Using MLP controller)
import numpy as np
import pandas as pd
import pickle
import sys, os
sys.path.insert(0, 'evoman')
from custom_players import player_0
from evoman.environment import Environment
experiment_name = 'MLP_SIM_0'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)
    
def run(genes, norm_inputs, en=1):
    
    env = Environment(experiment_name=experiment_name,
    				  playermode="ai",
    				  player_controller=player_0(normalize=norm_inputs),
    			  	  speed="normal",
    				  enemymode="static",
    				  level=2,
                      randomini="no",
    				  enemies=[en])
    return env.play(genes)




# with open("en2_full", "rb") as f:
#     res = pickle.load(f)
    
# pop = res["meta"]["last_gen"]

# en = res["meta"]["enemy"]

# elite = [_ for _ in pop if _["fitness"] == max([_["fitness"] for _ in pop]) ][0]

# run(elite, norm_inputs=True,en=en)


## The Generalist

elite_gene = {'bias': np.array([[-0.09283322,  0.        ,  0.98520709,  1.1762356 ,  1.        ]]),
 'weights': np.array([[ 0.48009632, -1.        ,  3.10544633,  2.129581  ,  1.        ],
        [-1.36082724, -2.5611546 ,  0.4127393 , -0.74149916, -0.49658297],
        [-1.23255571,  0.37160805,  0.27877869,  1.12885595,  0.22475044],
        [ 0.95180203, -1.46328255, -1.819437  ,  1.25725092,  0.26828811],
        [ 0.56662978,  0.23595821,  1.60559807,  0.61439899,  0.04863775],
        [ 0.04671857, -0.74785791,  0.4509434 ,  0.41540645,  0.46722509],
        [-1.05979488,  2.71512231, -1.14869503,  1.49700028, -0.55639558],
        [-2.0363591 ,  0.37399172, -0.64985232,  1.0080388 ,  1.62062319],
        [-2.60251272,  0.17092227, -0.88308242, -0.87927679, -1.93263948],
        [-1.67570154, -1.06402416,  1.78367449,  0.43027936,  2.19527802],
        [-0.06530854, -0.46360038, -1.15418033,  0.35748271, -0.94741732],
        [ 0.2430198 ,  0.54658942,  0.62452828,  0.78395332, -0.6268853 ],
        [-0.32483092, -1.32335592,  0.9003134 ,  0.78239987,  0.24827349],
        [-0.03353435,  0.96316892, -0.33677606,  2.88964576, -0.36491094],
        [ 1.32550035, -0.051698  ,  0.20941588,  0.19421073, -0.77097035],
        [ 0.38257077,  0.09615665,  1.73659944,  0.06748375,  1.23608023],
        [-1.62878901,  0.25563606, -0.30365525,  0.1766595 , -0.09366436],
        [ 1.3492051 ,  1.91819551,  1.28465372,  0.58985912,  0.08254211],
        [ 1.53319908,  0.04930124, -0.62941703, -0.72768493,  1.59189193],
        [-0.13660272,  0.31179246, -0.93962043,  0.42689033,  2.66491692]]),
 'parents': [70, 70],
 'ID': 53,
 'avg': 89.98739204866934,
 'std': 1.3093023463757574,
 'fitness': 88.67808970229358}


for en in [2,5,7,8]:
    run(elite_gene, norm_inputs=True, en=en)
    
    
    
    
    
    
    
    
    
    
    