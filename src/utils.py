from pathlib import Path  
import numpy as np
import pandas as pd
from numpy import exp as exp

from scipy.stats import binom,gmean

from datetime import date


def sort_by_value(dictionary, reverse = False):
    return [(k,dictionary[k]) for k in sorted(dictionary ,key= dictionary.get ,reverse=reverse)]

def lists_to_dictionary(list_of_lists):
    my_dict = [{} for j in range(len(list_of_lists))]

    for j, item in enumerate(list_of_lists):
        for i, rank in enumerate(item):
            my_dict[j][f'{i+1}'] = len(item)-rank+1
    return my_dict

def item_universe(list_of_dictionary):
    # which are the elements that we are ranking?
    # probably not necessray
    return list(frozenset().union(*[list(x.keys()) for x in list_of_dictionary]))

def first_order_marginals(list_of_dictionary):
    """
    Computes m_ik, the fraction of rankers that ranks item i as their kth choice
    (see Ammar and Shah, "Efficient Rank Aggregation Using Partial Data").  Works
    with either full or partial lists.
    """
    # get list of all the items
    all_items = item_universe(list_of_dictionary)
    # dictionaries for creating the matrix
    item_mapping(all_items)
    # create the m_ik matrix and fill it in
    m_ik = zeros((len(all_items),len(all_items)))
    n_r = len(list_of_dictionary)
    for r in list_of_dictionary:
        for item in r:
            m_ik[self.itemToIndex[item],r[item]-1] += 1
    return m_ik/n_r


def convert_to_ranks(scoreDict):
    """
    Accepts an input dictionary in which they keys are items to be ranked (numerical/string/etc.)
    and the values are scores, in which a higher score is better.  Returns a dictionary of
    items and ranks, ranks in the range 1,...,n.
    """
    # default sort direction is ascending, so reverse (see sort_by_value docs)
    x = sort_by_value(scoreDict,True)
    y = list(zip(list(zip(*x))[0],range(1,len(x)+1)))
    ranks = {}
    for t in y:
        ranks[t[0]] = t[1]
    return ranks


def item_ranks(rank_list):
    """
    Accepts an input list of ranks (each item in the list is a dictionary of item:rank pairs)
    and returns a dictionary keyed on item, with value the list of ranks the item obtained
    across all entire list of ranks.
    """
    item_ranks = {}.fromkeys(rank_list[0])
    for k in item_ranks:
        item_ranks[k] = [x[k] for x in rank_list]
    return item_ranks


def item_mapping(items):
    """
    Some methods need to do numerical work on arrays rather than directly using dictionaries.
    This function maps a list of items (they can be strings, ints, whatever) into 0,...,len(items).
    Both forward and reverse dictionaries are created and stored.
    """
    itemToIndex = {}
    indexToItem = {}
    next = 0
    for i in items:
        itemToIndex[i] = next
        indexToItem[next] = i
        next += 1
    return itemToIndex,indexToItem

def csv_to_dict(data):
    dictionary = []
    dic = data.to_dict()
    for key in dic.keys():
        dictionary.append(dic[key])
    return dictionary




def convert_dictionary(which_dataset):
    
    # take the original orderings and transform them the other way round + save it
    
    datapath = '../data/'
    dic = pd.read_csv(datapath + which_dataset + '.csv', index_col = 0)
    
    names = list(dic.index)

    dic_reversed = pd.DataFrame()
    for c_name in dic.columns:
        other_way = []
        for i in range(dic.shape[0]+1):
            if i in dic[c_name].values:
                other_way.append(dic[c_name].tolist().index(i))
        other_way_names = []
        for i in other_way:
            other_way_names.append(names[i])
        dic_reversed[c_name] = other_way_names
        
    dic_reversed.to_csv(f'{datapath}{which_dataset}_Reversed.csv')
        
def transforForMANI(data):
    
    # funzionava su funzionaTutto.ipnb
    _data = pd.read_csv(f'../data/{data}.csv', index_col = 0)    
    
    df = pd.DataFrame()
    for i in range(_data.shape[1]):
        arr = [0 for r in range(_data.shape[0])]
        for j in range(_data.shape[0]):
            arr[j] = _data.loc[j]
            print(i)
        
