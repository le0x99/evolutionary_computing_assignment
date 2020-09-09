#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  5 16:58:38 2020

@author: leonarddariusvorbeck
"""

## First Simulation (Using MLP controller)
import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, 'evoman')
from custom_players import player_0
from evoman.environment import Environment
experiment_name = 'MLP_SIM_0'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

os.environ["SDL_VIDEODRIVER"] = "dummy"


# initializes environment and tests the individual

def run(genes, norm_inputs, en=1):
    env = Environment(experiment_name=experiment_name,
    				  playermode="ai",
    				  player_controller=player_0(normalize=norm_inputs),
    			  	  speed="fastest",
    				  enemymode="static",
    				  level=2,
                      randomini="no",
    				  enemies=[en])
    return env.play(genes)

## CrossOver Mechanism
def crossover(parent1, parent2,
              gene_sets = ["bias", "weights" ],
              m0=.05,
              m1=.1,
              m2=.05,
              m3=.05,
              m4=.05):
    child = dict()
    for gene_set in gene_sets:
        # Copy the gene set of one parent as a placeholder
        child[gene_set] = copy(parent1[gene_set])
        # Iterate through the items of each gene set
        for i in range(len(child[gene_set][0])):
            # 50% chance of coming from either of one parent
            choice = copy(np.random.choice([parent1[gene_set][0][i],
                                       parent2[gene_set][0][i]]))
            child[gene_set][0][i] = choice
            ## Mutations (Notr : weights of hidden neuron 0 are stored in child["weights_1"][:, 0])
            # Mutation 0 - weight becomes the mean of the parents (perfect mixture)
            if np.random.binomial(1, m0) == 1:
                child[gene_set][0][i] = (parent2[gene_set][0][i] + parent1[gene_set][0][i])/2
            # Mutation 1 - Add a little bit of noise (imperfect copy)
            if np.random.binomial(1, m1) == 1:
                child[gene_set][0][i] = child[gene_set][0][i] + np.random.normal(1, .5)
            # Mutation 2 - Weight gets 0 (Quasi-Deletion)
            if np.random.binomial(1, m2) == 1:
                child[gene_set][0][i] = 0
            # Mutation 3 - Swap Coefficient
            if np.random.binomial(1, m3) == 1:
                child[gene_set][0][i] = - child[gene_set][0][i] 
            # Mutation 4 - weight gets 1
            if np.random.binomial(1, m4) == 1:
                child[gene_set][0][i] = 1
                
    child["parents"] = [ parent1["ID"] + parent2["ID"], parent2["ID"] + parent1["ID"] ]
    child["parents"].sort()
    return child

def test_pop(pop,enemy):
    for individual in pop:
        fitness, player_life, enemy_life, time_exp = run(individual,
                                                         norm_inputs=True, en=enemy)
        individual["fitness"] = fitness
        individual["player_life"] = player_life
        individual["enemy_life"] = enemy_life
        individual["time_exp"] = time_exp
    return pop

def reproduce(pop, highf_r, lowf_r,
              m0,m1,m2,m3,m4,
              keep_elite=True):
    npop = len(pop)
    pop_ = pd.DataFrame(pop) if type(pop)!= pd.DataFrame else pop
    elite = pop_[pop_["fitness"] >= pop_["fitness"].quantile(.9)].to_dict("records")
    above_avg = pop_[pop_["fitness"] > pop_["fitness"].mean()]
    below_avg = pop_[pop_["fitness"] < pop_["fitness"].mean()]
    breed_pool = above_avg.sample(frac=highf_r).to_dict("records") + below_avg.sample(frac=lowf_r).to_dict("records")
    new_gen = []
    while len(new_gen) < npop - len(elite):
        p1 = np.random.choice(breed_pool)
        p2 = np.random.choice(breed_pool)
        if p1["ID"] == p2["ID"] or p1["parents"] == p2["parents"]:
            continue
        child = crossover(p1,p2,["bias", "weights" ],m0,m1,m2,m3,m4)
        child["ID"] = len(new_gen)
        new_gen.append(child)
    if keep_elite:
        return  new_gen + elite
    else:
        return new_gen

import datetime
from copy import deepcopy as copy
def run_sim(controller=player_0(normalize=True),
            enemy=1,
            npop=50,
            ngens=10,
            m0=.07,
            m1=.1,
            m2=.07,
            m3=.07,
            m4=.07,
            highf_r = 1.,
            lowf_r = .33,
            n_hidden=None,
            keep_elite=True
            ):
    generations = {}
    for gen in range(ngens):
        print(" - - - - GENERATION %s - - - - " % gen)
        if gen == 0:
            pop = [ {"bias" : np.random.randn(1, 5),
                     "weights" : np.random.randn(20, 5),
                     "ID" : _ , "parents" : _} for _ in range(npop)]
            pop = test_pop(pop,enemy)
            generations[gen] = pop
        else:
            new_pop = reproduce(generations[gen-1],
                            highf_r,lowf_r,
                            m0,m1,m2,m3,m4,
                            keep_elite)
            new_pop = test_pop(new_pop,enemy)
            generations[gen] = new_pop
        
    FF = []
    for gen in generations:
        pop = generations[gen]
        pop_fitness = [_["fitness"] for _ in pop]
        FF.append(pop_fitness)
    
    df = pd.DataFrame(FF).T
    result = {"data" : df, "meta" : {"npop" : npop, "ngens" :ngens, "m3" : m3,"m4":m4,
                                     "m0" : m0, "m1" : m1, "m2":m2,"highf_r" : highf_r,
                                     "lowf_r":lowf_r, "n_hidden":n_hidden,
                                     "elite":keep_elite, "enemy" : enemy}}
    
    with open(str(datetime.datetime.now()), "wb") as f:
        pickle.dump(result, f)
        
    return df
    
import pickle
        
# Result analysis
df = run_sim(controller=player_0(normalize=True),
            enemy=1,
            npop=100,
            ngens=15,
            m0=.12,
            m1=.2,
            m2=.12,
            m3=.15,
            m4=.12,
            highf_r = 1.,
            lowf_r = .05,
            n_hidden=None,
            keep_elite=True
            )
results = []
for e in [1,2,3,4,5,6,7,8]:
    df = run_sim(controller=player_0(normalize=True),
            enemy=e,
            npop=100,
            ngens=20,
            m0=.12,
            m1=.2,
            m2=.12,
            m3=.15,
            m4=.12,
            highf_r = 1.,
            lowf_r = .05,
            n_hidden=None,
            keep_elite=True
            )
    results.append(df)
    






# pop = [ {"bias_hidden" : np.random.randn(1, n_hidden),
#          "bias_output" : np.random.randn(1, 5),
#          "weights_1" : np.random.randn(20, n_hidden),
#          "weights_2" : np.random.randn(n_hidden, 5),
#          "ID" : _ , "parents" : _} for _ in range(npop)]
# pop = test_pop(pop)

# new_pop = reproduce(pop,
#                             highf_r,lowf_r,
#                             m0,m1,m2,m3,m4,
#                             keep_elite)
# new_pop = test_pop(new_pop)

# new_pop1 = reproduce(new_pop,
#                             highf_r,lowf_r,
#                             m0,m1,m2,m3,m4,
#                             keep_elite)
# new_pop1 = test_pop(new_pop1)


# [_["fitness"] for _ in new_pop ]

