from graphs import *
from graph_functions import *
import aggregators_FAIR 
import aggregators_NONFAIR
import aggregators_OURS 

import numpy as np
import pandas as pd
import os
import sys

import networkx as nx

import time
from datetime import date
today = str(date.today()).replace('-', '')
pd.set_option('display.max_rows', None)

import pickle
import json

sys.path.append('../RankingAggregation/MC4/')
from mc4.algorithm import mc4_aggregator


from metrics import *

def get_results(data, where):
    
    path_data = '../data/' + where + '/'
    path_results = '../results/' + where + '/' + data + '/'
        
    dic = pd.read_csv(path_data + data + '.csv', index_col = 0)
    print(path_data + data)

    g = mc4_aggregator(path_data + data + '.csv', header_row = 0, index_col = 0)
    json.dump( g, open(path_results + 'mc4_aggregator.json', 'w' ))
    
    g = aggregators_NONFAIR.borda_aggregation(dic)
    json.dump( g, open(path_results + 'borda_aggregation.json', 'w' ))

#     g = aggregators_NONFAIR.median_aggregation(dic)
#     json.dump( g, open(path_results + 'median_aggregation.json', 'w' ))
    
    g = aggregators_NONFAIR.highest_rank(dic)
    json.dump( g, open(path_results + 'highest_rank.json', 'w' ))
    
    g= aggregators_NONFAIR.lowest_rank(dic)
    json.dump( g, open(path_results + 'lowest_rank.json', 'w' ))
    
    g = aggregators_NONFAIR.stability_selection(dic),
    json.dump( g, open(path_results + 'stability_selection.json', 'w' ))
    
    g = aggregators_NONFAIR.exponential_weighting(dic),
    json.dump( g, open(path_results + 'exponential_weighting.json', 'w' ))
    
    g = aggregators_NONFAIR.stability_enhanced_borda(dic),
    json.dump( g, open(path_results + 'stability_enhanced_borda.json', 'w' ))
    
    g = aggregators_NONFAIR.exponential_enhanced_borda(dic),
    json.dump( g, open(path_results + 'exponential_enhanced_borda.json', 'w' ))
    
    g = aggregators_NONFAIR.robust_aggregation(dic),
    json.dump( g, open(path_results + 'robust_aggregation.json', 'w' ))
    
    g = aggregators_NONFAIR.round_robin(dic),
    json.dump( g, open(path_results + 'round_robin.json', 'w' ))
    
    g = aggregators_FAIR.bestRankAggregated(dic),
    json.dump( g, open(path_results + 'bestRankAggregated.json', 'w' ))
    
    
def get_resultsMC(data, where):
    
    path_data = '../data/' + where + '/'
    path_results = '../results/' + where + '/' + data + '/MC/'
    if os.path.exists(path_results) == False:
        os.mkdir(path_results)  
    
    dic = pd.read_csv(path_data + data + '.csv', index_col = 0)
    print(path_data + data)

        
    alpha = 1

    G1 = MC1(data, where)
    G2 = MC2(data, where)
    G3 = MC3(data, where)
    
    S = [False for i in range(6)]
    alphas = [-1 for i in range(6)]
    
    for alpha in np.arange(1, 0, -0.05):
        
        print('I am trying alpha =', alpha)
        
        G1_erg = makeErgodic(G1, alpha)
        G2_erg = makeErgodic(G2, alpha)
        G3_erg = makeErgodic(G3, alpha)

        G1_ergFair = makeErgodic(makeFair(G1), alpha)
        G2_ergFair = makeErgodic(makeFair(G2), alpha)
        G3_ergFair = makeErgodic(makeFair(G3), alpha)

        # these are all graphs, made ergodic

        print(f'G1: is the method working with {alpha}?', check_ifProbDist(get_statDistr(G1_erg)))
        if check_ifProbDist(get_statDistr(G1_erg)) == True and S[0] == False:
            results = {'MC1': aggregators_NONFAIR.MC1_getRanking(data, alpha, where)}
            results_df = pd.DataFrame.from_dict(results)
            results_df = results_df.sort_index()
            results_df.to_csv(path_results  + f'MC1results.csv')
            S[0] = True
            alphas[0] = alpha.round(2)
            
        print(f'G2: is the method working with {alpha}?', check_ifProbDist(get_statDistr(G2_erg)))
        if check_ifProbDist(get_statDistr(G2_erg)) == True and S[1] == False:
            results = {'MC2': aggregators_NONFAIR.MC2_getRanking(data, alpha, where)}
            results_df = pd.DataFrame.from_dict(results)
            results_df = results_df.sort_index()
            results_df.to_csv(path_results  + f'MC2results.csv')
            S[1] = True
            alphas[1] = alpha.round(2)
            
        print(f'G3: is the method working with {alpha}?', check_ifProbDist(get_statDistr(G3_erg)))
        if check_ifProbDist(get_statDistr(G3_erg)) == True and S[2] == False:
            results = {'MC3': aggregators_NONFAIR.MC3_getRanking(data, alpha, where)}
            results_df = pd.DataFrame.from_dict(results)
            results_df = results_df.sort_index()
            results_df.to_csv(path_results  + f'MC3results.csv')
            S[2] = True
            alphas[2] = alpha.round(2)
        
        print(f'G1fair: is the method working with {alpha}?', check_ifProbDist(get_statDistr(G1_ergFair)))
        if check_ifProbDist(get_statDistr(G1_ergFair)) == True and S[3] == False: 
            resultsFAIR = {'MC1Fair': aggregators_OURS.MC1_getRankingFair(data, alpha, where)}
            results_dfFAIR = pd.DataFrame.from_dict(resultsFAIR)
            results_dfFAIR = results_dfFAIR.sort_index()
            results_dfFAIR.to_csv(path_results + f'MC1FAIRresults.csv')
            S[3] = True
            alphas[3] = alpha.round(2)
            
        
        print(f'G2fair: is the method working with {alpha}?', check_ifProbDist(get_statDistr(G2_ergFair)))
        if check_ifProbDist(get_statDistr(G2_ergFair)) == True and S[4] == False:
            resultsFAIR = {'MC2Fair': aggregators_OURS.MC2_getRankingFair(data, alpha, where)}
            results_dfFAIR = pd.DataFrame.from_dict(resultsFAIR)
            results_dfFAIR = results_dfFAIR.sort_index()
            results_dfFAIR.to_csv(path_results + f'MC2FAIRresults.csv')
            S[4] = True
            alphas[4] = alpha.round(2)

        print(f'G3fair: is the method working with {alpha}?', check_ifProbDist(get_statDistr(G3_ergFair)))
        if check_ifProbDist(get_statDistr(G3_ergFair)) == True and S[5] == False:
            resultsFAIR = {'MC3Fair': aggregators_OURS.MC3_getRankingFair(data, alpha, where)}
            results_dfFAIR = pd.DataFrame.from_dict(resultsFAIR)
            results_dfFAIR = results_dfFAIR.sort_index()
            results_dfFAIR.to_csv(path_results + f'MC3FAIRresults.csv')
            S[5] = True
            alphas[5] = alpha.round(2)
        
        alphas = np.asarray(alphas)
        np.savetxt(path_results + 'alphas.txt', alphas.round(2))

        
        if S == [True for i in range(3)]:
            break


def aggregateResults(data, where):

    path_results = '../results/' + where + '/' + data+ '/' 

    methods = []
    for file in os.listdir(path_results):
        if 'json' in file:
            methods.append(str(file).replace(".json",""))

    for file in os.listdir(path_results + '/MC/'):
        if 'MC' in file:
            methods.append(str(file).replace("results.csv",""))


    my_dict = {}
    results =[]

    for m in methods:
        if 'MC' not in m:
            with open(path_results + f'{m}.json', "r") as d:
                s = json.load(d)
            if type(s) == list:
                s = s[0]
            item = [m,s]
            results.append(item)
        for item in results:
            my_dict[item[0]] = item[1]

        if 'MC' in m:
            result = pd.read_csv(path_results + '/MC/' + m + 'results.csv', index_col = 0).to_dict()
            my_dict.update(result)
    results_df = pd.DataFrame.from_dict(my_dict)
    results_df = results_df.sort_index()
    results_df.to_csv(path_results + '_RESULTS.csv')
    
    

def evaluate_data(data, where):
    path_results = f'../results/{where}/{data}/'
    path_data = f'../data/{where}/'
    _data = pd.read_csv(path_data + data + '.csv', index_col = 0)   
    
    if where == 'CACHELdataMallows/transf' or where == 'realWorldData':
        _data = _data + 1
    if where == 'RealWorldData':
        
        _data = pd.read_csv(path_data + data + '.csv', index_col = 0)
    
        
    rankers = list(_data.columns)
    fairness_rankers = []
    exposure_rankers = []
    kemeny_distance = []
    levels = [int(10*np.shape(_data)[1]/100), int(20*np.shape(_data)[1]/100), int(50*np.shape(_data)[1]/100)]
    if where == 'RealWorldData':
        levels = [int(10*np.shape(_data)[0]/100), int(20*np.shape(_data)[0]/100), int(50*np.shape(_data)[0]/100)]
    for k in levels:
        fairness_rankers.append(fairness_loss(_data, k))
    exposure_rankers = group_exposure(_data)
    R_par_original = R_par(_data)

    
    dictionary = dict()
    
    dictionary = {# rankers analysis
                  'fairness_rankers@10' : fairness_rankers[0],
                  'fairness_rankers@20' : fairness_rankers[1],
                  'fairness_rankers@50' : fairness_rankers[2],
                  'exposure_rankers' : exposure_rankers,
                  'R_par_rankers' : R_par_original}

#     with open(path_results + '_fairness_DATA.txt', "w") as d:
#         json.dump(dictionary, d) 


    results_df = pd.DataFrame.from_dict(dictionary)
    results_df.index = rankers
    results_df.to_csv(path_results + '_fairness_DATA.csv')


        
def evaluate_results(data, where):
    
    path_data = f'../data/{where}/'
    path_results = f'../results/{where}/{data}/'
    
    if where == 'RealWorldData':
        _data = pd.read_csv(path_data + data + '.csv', index_col = 0) + 1
    else:
        _data = pd.read_csv(path_data + data + '.csv', index_col = 0)
    _data_results = pd.read_csv(path_results + '_RESULTS.csv', index_col = 0)
    methods = list(_data_results.columns)

    fairness_aggregation = []
    exposure_aggregation = []
    kemeny_distance = []
    
    levels = [int(10*np.shape(_data)[1]/100), int(20*np.shape(_data)[1]/100), int(50*np.shape(_data)[1]/100)]
    if where == 'RealWorldData':
        levels = [int(10*np.shape(_data)[0]/100), int(20*np.shape(_data)[0]/100), int(50*np.shape(_data)[0]/100)]    
    for k in levels:
        fairness_aggregation.append(fairness_loss(_data_results, k))
    exposure_aggregation = group_exposure(_data_results)
    kemeny_distance = kemeny_distances(_data_results, _data)
    R_par_results = R_par(_data_results)
    consACC = calcConsensusAccuracy(_data_results, _data)
    topK = topKpar(_data_results)
    
    dictionary = dict()
    
    dictionary = {# aggregation analysis
                  'fairness_aggregation@10' : fairness_aggregation[0],
                  'fairness_aggregation@20' : fairness_aggregation[1],
                  'fairness_aggregation@50' : fairness_aggregation[2],
                  'exposure_aggregation' : exposure_aggregation,
                  'R_par_results' : R_par_results,
                  'kemeny_distance' : kemeny_distance,
                  'consensus accuracy' : consACC,
                  'topKparity': topK}

        
    results_df = pd.DataFrame.from_dict(dictionary)
    results_df.index = methods
    results_df.to_csv(path_results + '_fairness_RESULTS.csv')
    print('I workd')
#             alpha -= 0.01


def evaluate_resultCACHEL(data, where):
    
    path_data = f'../data/{where}/'
    path_results = f'../results/{where}/{data}/'
    for file in os.listdir(path_results):
        if 'CACHELresultsTransformed_' in file:
            dataTransf = file 
    else:
        _data = pd.read_csv(path_data + data + '.csv', index_col = 0)
        
    _data_results = pd.read_csv(path_results + dataTransf, index_col = 0)
    methods = list(_data_results.columns)

    fairness_aggregation = []
    exposure_aggregation = []
    kemeny_distance = []
    
    levels = [int(10*np.shape(_data)[1]/100), int(20*np.shape(_data)[1]/100), int(50*np.shape(_data)[1]/100)]
    if where == 'RealWorldData':
        levels = [int(10*np.shape(_data)[0]/100), int(20*np.shape(_data)[0]/100), int(50*np.shape(_data)[0]/100)]    
    for k in levels:
        fairness_aggregation.append(fairness_loss(_data_results, k))
    exposure_aggregation = group_exposure(_data_results)
    kemeny_distance = kemeny_distances(_data_results, _data)
    R_par_results = R_par(_data_results)
    consACC = calcConsensusAccuracy(_data_results, _data)
    topK = topKpar(_data_results)
    
    dictionary = dict()
    
    dictionary = {# aggregation analysis
                  'fairness_aggregation@10' : fairness_aggregation[0],
                  'fairness_aggregation@20' : fairness_aggregation[1],
                  'fairness_aggregation@50' : fairness_aggregation[2],
                  'exposure_aggregation' : exposure_aggregation,
                  'R_par_results' : R_par_results,
                  'kemeny_distance' : kemeny_distance,
                  'consensus accuracy' : consACC,
                  'topKparity': topK}

        
    results_df = pd.DataFrame.from_dict(dictionary)
    results_df.index = methods
    results_df.to_csv(path_results + '_fairness_RESULTSCACHEL.csv')
#             alpha -= 0.01




