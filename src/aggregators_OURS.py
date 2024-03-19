from numpy.random import permutation
import utils
from numpy import exp as exp
import numpy as np
from scipy.stats import binom
from assignment import linear_assignment

from utils import *
from graph_functions import *
from graphs import *

def MC1_getRankingFair(data, alpha, where):
    print('MC1_getRankingFair')
    ergodic_matrix = makeErgodic(MC1(data, where), alpha)
    ergodic_matrix_fair = makeFair(ergodic_matrix)
    ranking = list(fromScoresToRank(get_statDistr(ergodic_matrix_fair)))
    
    path_data = '../data/' + where + '/'
    dic = pd.read_csv(path_data + data + '.csv', index_col = 0)
    ranklist = csv_to_dict(dic)
    
    aggRanks = {}.fromkeys(ranklist[0])
    
    for key in aggRanks:
        if ranking:
            aggRanks[key] = ranking.pop(0)
        else:
            break
        
    return aggRanks


def MC2_getRankingFair(data, alpha, where):
    print('MC2_getRankingFair')
    ergodic_matrix = makeErgodic(MC2(data, where), alpha)
    ergodic_matrix_fair = makeFair(ergodic_matrix)
    ranking = list(fromScoresToRank(get_statDistr(ergodic_matrix_fair)))
    
    path_data = '../data/' + where + '/'
    dic = pd.read_csv(path_data + data + '.csv', index_col = 0)
    ranklist = csv_to_dict(dic)
    
    aggRanks = {}.fromkeys(ranklist[0])
    
    for key in aggRanks:
        if ranking:
            aggRanks[key] = ranking.pop(0)
        else:
            break
        
    return aggRanks


def MC3_getRankingFair(data, alpha, where):
    print('MC3_getRankingFair')

    ergodic_matrix = makeErgodic(MC3(data, where), alpha)
    ergodic_matrix_fair = makeFair(ergodic_matrix)
    ranking = list(fromScoresToRank(get_statDistr(ergodic_matrix_fair)))
    
    path_data = '../data/' + where + '/'
    dic = pd.read_csv(path_data + data + '.csv', index_col = 0)
    ranklist = csv_to_dict(dic)
    
    aggRanks = {}.fromkeys(ranklist[0])
    
    for key in aggRanks:
        if ranking:
            aggRanks[key] = ranking.pop(0)
        else:
            break
        
    return aggRanks


def MC4_getRankingFair(data, alpha, where):
    ergodic_matrix = makeErgodic(method1(data, where), alpha)
    ergodic_matrix_fair = makeFair(ergodic_matrix)
    ranking = list(fromScoresToRank(get_statDistr(ergodic_matrix_fair)))
    
    path_data = '../data/' + where + '/'
    dic = pd.read_csv(path_data + data + '.csv', index_col = 0)
    ranklist = csv_to_dict(dic)
    
    aggRanks = {}.fromkeys(ranklist[0])
    
    for key in aggRanks:
        if ranking:
            aggRanks[key] = ranking.pop(0)
        else:
            break
        
    return aggRanks


def MC5_getRankingFair(data, alpha, where):
    ergodic_matrix = makeErgodic(method2(data, where), alpha)
    ergodic_matrix_fair = makeFair(ergodic_matrix)
    ranking = list(fromScoresToRank(get_statDistr(ergodic_matrix_fair)))
    
    path_data = '../data/' + where + '/'
    dic = pd.read_csv(path_data + data + '.csv', index_col = 0)
    ranklist = csv_to_dict(dic)
    
    aggRanks = {}.fromkeys(ranklist[0])
    
    for key in aggRanks:
        if ranking:
            aggRanks[key] = ranking.pop(0)
        else:
            break
        
    return aggRanks

