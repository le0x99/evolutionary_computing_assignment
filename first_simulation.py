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

os.environ["SDL_VIDEODRIVER"] = "dummy"


# initializes environment and tests the individual

def run(genes, norm_inputs, en=1):
    env = Environment(experiment_name=experiment_name,
    				  playermode="ai",
    				  player_controller=player_1(normalize=norm_inputs),
    			  	  speed="fastest",
    				  enemymode="static",
    				  level=2,
                      randomini="no",
    				  enemies=[en])
    return env.play(genes)

## First run - Try to find a good base geneset using a large initial population.

# # Initialize large random population
# npop = 200
# n_hidden = 4
# pop = [ {"bias_hidden" : np.random.randn(1, n_hidden),
#          "bias_output" : np.random.randn(1, 5),
#          "weights_1" : np.random.randn(20, n_hidden),
#          "weights_2" : np.random.randn(n_hidden, 5),
#          "ID" : _ } for _ in range(npop)]

# # Test them

# for individual in pop:
#     fitness, player_life, enemy_life, time_exp = run(individual,
#                                                      norm_inputs=True, en=1)
#     individual["fitness"] = fitness
#     individual["player_life"] = player_life
#     individual["enemy_life"] = enemy_life
#     individual["time_exp"] = time_exp

    
# gen0 = pd.DataFrame(pop)

# # Show fitness dist of the init-population
# gen0.fitness.hist(bins=50)



## CrossOver Mechanism
def crossover(parent1, parent2,
              c=.2,
              m0=.1,
              m1=.2,
              m2=.1):
    child = dict()
    if np.random.binomial(1, c) == 1:
        child = np.random.choice([parent1, parent2])
        return child
                
    for gene_set in ["bias_hidden", "bias_output", "weights_1", "weights_2" ]:
        # Copy the gene set of one parent as a placeholder
        child[gene_set] = parent1[gene_set]
        # Iterate through the items of each gene set
        for i in range(len(child[gene_set][0])):
            # 50% chance of coming from either of one parent
            choice = np.random.choice([parent1[gene_set][0][i],
                                       parent2[gene_set][0][i]])
            child[gene_set][0][i] = choice
            ## Mutations (Notr : weights of hidden neuron 0 are stored in child["weights_1"][:, 0])
            # Mutation 0 - weight becomes the mean of the parents (perfect mixture)
            if np.random.binomial(1, m0) == 1:
                child[gene_set][0][i] = (parent2[gene_set][0][i] + parent1[gene_set][0][i])/2
            # Mutation 1 - Add a little bit of noise (imperfect copy)
            if np.random.binomial(1, m1) == 1:
                child[gene_set][0][i] = child[gene_set][0][i] + np.random.normal(0, .002)
            # Mutation 2 - Weight gets 0 (Quasi-Deletion)
            if np.random.binomial(1, m2) == 1:
                child[gene_set][0][i] = 0
    return child

# ## Reproduction
# # Above average fitness individuals
# above_avg = gen0[gen0.fitness > gen0.fitness.mean()]
# # Below average fitness individuals
# below_avg = gen0[gen0.fitness < gen0.fitness.mean()]
# # For reproduction 100% of above average are used and 20% of below average
# highf_r = 1.
# lowf_r = .1
# breed_pool = above_avg.sample(frac=highf_r).to_dict("records") + below_avg.sample(frac=lowf_r).to_dict("records")

# #Mutation settings
# m0=.1,
# m1=.33,
# m2=.05



# new_gen = []
# while len(new_gen) < npop:
#     p1 = np.random.choice(breed_pool)
#     p2 = np.random.choice(breed_pool)
#     while p1["ID"] == p2["ID"]:
#         p2 = np.random.choice(breed_pool)
#     child = crossover(p1, p2, m0,m1,m2)
#     child["ID"] = len(new_gen)
#     new_gen.append(child)
    
# # Test new breed
# for individual in new_gen:
#     fitness, player_life, enemy_life, time_exp = run(individual,
#                                                      norm_inputs=True, en=1)
#     individual["fitness"] = fitness
#     individual["player_life"] = player_life
#     individual["enemy_life"] = enemy_life
#     individual["time_exp"] = time_exp
    
# gen1 = pd.DataFrame(new_gen)


# gen01fitness.hist(bins=50)

def test_pop(pop):
    for individual in pop:
        fitness, player_life, enemy_life, time_exp = run(individual,
                                                         norm_inputs=True, en=1)
        individual["fitness"] = fitness
        individual["player_life"] = player_life
        individual["enemy_life"] = enemy_life
        individual["time_exp"] = time_exp
    return 

def reproduce(pop, highf_r, lowf_r,
              c, m0,m1,m2,
              npop,
              keep_elite, elite_quantile,
              strict_selection):
    pop_ = pd.DataFrame(pop) if type(pop)!= pd.DataFrame else pop
    above_avg = pop_[pop_["fitness"] > pop_["fitness"].mean()]
    below_avg = pop_[pop_["fitness"] < pop_["fitness"].mean()]
    if strict_selection:
        breed_pool = above_avg.sample(frac=highf_r).to_dict("records") + below_avg.sample(frac=lowf_r).to_dict("records")
    else:
        breed_pool = pop_.to_dict("records")
    new_gen = []
    while len(new_gen) < npop:
        p1 = np.random.choice(breed_pool)
        p2 = np.random.choice(breed_pool)
        while p1["ID"] == p2["ID"]:
            p2 = np.random.choice(breed_pool)
        child = crossover(p1, p2,c, m0,m1,m2)
        child["ID"] = len(new_gen)
        new_gen.append(child)
    if keep_elite:
        elite = pop_[pop_["fitness"] >= pop_["fitness"].quantile(elite_quantile)].to_dict("records")
        new_gen = new_gen + elite
    return  list(np.random.choice(new_gen, npop))

import datetime

def run_sim(controller=player_1(normalize=True),
            npop=100,
            ngens=20,
            c=.2,
            m0=.1,
            m1=.1,
            m2=.05,
            highf_r = 1.,
            lowf_r = .5,
            n_hidden=4,
            keep_elite=True,
            elite_quantile=.9,
            strict_selection=True
            ):
    generations = {}
    for gen in range(ngens):
        print(" - - - - GENERATION %s - - - - " % gen)
        if gen == 0:
            pop = [ {"bias_hidden" : np.random.randn(1, n_hidden),
                     "bias_output" : np.random.randn(1, 5),
                     "weights_1" : np.random.randn(20, n_hidden),
                     "weights_2" : np.random.randn(n_hidden, 5),
                     "ID" : _ } for _ in range(npop)]
            test_pop(pop)
        else:
            pop = reproduce(generations[gen-1], highf_r, lowf_r,c,m0,m1,m2,npop,
                            keep_elite, elite_quantile, strict_selection)
            test_pop(pop)
            
        generations[gen] = pop
        
    FF = []
    for gen in generations:
        pop = generations[gen]
        pop_fitness = [_["fitness"] for _ in pop]
        FF.append(pop_fitness)
    
    df = pd.DataFrame(FF).T
    result = {"data" : df, "meta" : {"npop" : npop, "ngens" :ngens, "c" : c,
                                     "m0" : m0, "m1" : m1, "m2":m2,"highf_r" : highf_r,
                                     "lowf_r":lowf_r, "n_hidden":n_hidden}}
    
    with open(str(datetime.datetime.now()), "wb") as f:
        pickle.dump(result, f)
        
    return df
    
import pickle
        
# Result analysis
df = run_sim(controller=player_1(normalize=True),
            npop=200,
            ngens=15,
            c=.2,
            m0=.3,
            m1=.3,
            m2=.05,
            highf_r = 1.,
            lowf_r = .5,
            n_hidden=5,
            keep_elite=True,
            elite_quantile=.7,
            strict_selection=False
            )



