from numpy.random import permutation
import utils
from numpy import exp as exp
import numpy as np
from scipy.stats import binom
from assignment import linear_assignment

from utils import *

def borda_aggregation(data):
    
    print('borda_aggregation')
    """
    Computes aggregate rank by Borda score.  For each item and list of ranks,
    compute:

            B_i(c) = # of candidates ranks BELOW c in ranks_i

    Then form:

            B(c) = sum(B_i(c))
    and sort in order of decreasing Borda score.

    The aggregate ranks are returned as a dictionary, as are in input ranks.
    """
    ranklist = csv_to_dict(data)
    aggRanks = {}.fromkeys(ranklist[0])
    for item in aggRanks:
        aggRanks[item] = 0

    maxRank = len(aggRanks)
    for r in ranklist:
        for item in r:
            aggRanks[item] += maxRank - r[item]
    return convert_to_ranks(aggRanks)

def median_aggregation(data):
    print('median_aggregation')
    """
    Computes median aggregate rank.  Start's each items score M_i at zero,
    and then for each rank 1,...,M, the item's score is incremented by the
    number of lists in which it has that rank.  The first item over L/2
    gets rank 1, the next rank 2, etc.  Ties are broken randomly.

    Aggregate ranks are returned as a dictionary.
    """
    rank_list = csv_to_dict(data)

    theta = 1.0*len(rank_list)/2

    aggRanks = {}.fromkeys(rank_list[0])

    for k in aggRanks:
        aggRanks[k] = 0

    # lists of item ranks (across all lists) for each item
    item_rankings = item_ranks(rank_list)

    # this holds the eventual voted ranks
    med_ranks = {}.fromkeys(rank_list[0])

    # the next rank that needs to be assigned

    next_rank = 1
    # once the next-to-last item has a rank, assign the one remaining item the last rank

    for r in range(1,len(med_ranks)):
        # increment scores
        for k in aggRanks:
            aggRanks[k] += item_rankings[k].count(r)
        # check if any of the items are over threshold; randomly permute
        #   all over-threshold items for rank assignment (tie breaking)
        items_over = list(permutation([k for k in aggRanks if aggRanks[k] >= theta]))
        for i in range(len(items_over)):
            med_ranks[items_over[i]] = next_rank + i
            aggRanks.pop(items_over[i])
        next_rank = next_rank + len(items_over)
        if next_rank == len(med_ranks):
            break
    # if we are out of the loop, there should only be one item left to rank
    med_ranks[list(aggRanks.keys())[0]] = len(med_ranks)
    return med_ranks





def highest_rank(data):
    print('highest_rank')

    """
    Each item is assigned the highest rank it obtains in all of the
    rank lists.  Ties are broken randomly.
    """
    
    rank_list = csv_to_dict(data)

    min_ranks = {}.fromkeys(rank_list[0])
    item_rankings = item_ranks(rank_list)
    for k in min_ranks:
        min_ranks[k] = min(item_rankings[k])
    # sort the highest ranks dictionary by value (ascending order)
    pairs = sort_by_value(min_ranks)
    # assign ranks in order
    pairs = list(zip(list(zip(*pairs))[0],range(1,len(item_rankings)+1)))
    # over-write the min_ranks dict with the aggregate ranks
    for (item,rank) in pairs:
        min_ranks[item] = rank
    return min_ranks


def lowest_rank(data):
    print('lowest_rank')

    """
    Each item is assigned the lowest rank it obtains in all of the rank
    lists.  Ties are broken randomly.
    """
    
    rank_list = csv_to_dict(data)

    max_ranks = {}.fromkeys(rank_list[0])
    item_rankings = item_ranks(rank_list)
    for k in max_ranks:
        max_ranks[k] = max(item_rankings[k])
    # sort the worst ranks dictionary by value (ascending order)
    pairs = sort_by_value(max_ranks)
    # assign ranks in order
    pairs = list(zip(list(zip(*pairs))[0],range(1,len(item_rankings)+1)))
    # over-write the max_ranks dict with the aggregate ranks
    for (item,rank) in pairs:
        max_ranks[item] = rank
    return max_ranks


def stability_selection(data,theta=None):
    print('stability_selection')
    """
    For every list in which an item is ranked equal to or higher than theta
    (so <= theta), it recieves one point.  Items are then ranked from most to
    least points and assigned ranks.  If theta = None, then it is set equal to
    half the number of items to rank.
    """
    
    rank_list = csv_to_dict(data)

    if theta is None:
        theta = 1.0*len(rank_list[0])/2
    scores = {}.fromkeys(rank_list[0])
    item_rankings = item_ranks(rank_list)
    for k in scores:
        scores[k] = sum([i <= theta for i in item_rankings[k]])
    return convert_to_ranks(scores)


def exponential_weighting(data,theta=None):
    print('exponential_weighting')
    """
    Like stability selection, except items are awarded points according to
    Exp(-r/theta), where r = rank and theta is a threshold.  If theta = None,
    then it is set equal to half the number of items to rank.
    """
    
    rank_list = csv_to_dict(data)

    if theta is None:
        theta = 1.0*len(rank_list[0])/2
    scores = {}.fromkeys(rank_list[0])
    item_rankings = item_ranks(rank_list)
    for k in scores:
        scores[k] = exp([-1.0*x/theta for x in item_rankings[k]]).sum()
    return convert_to_ranks(scores)


def stability_enhanced_borda(data,theta=None):
    print('stability_enhanced_borda')
    """
    For stability enhanced Borda, each item's Borda score is multiplied
    by its stability score and larger scores are assigned higher ranks.
    """
    
    rank_list = csv_to_dict(data)

    if theta is None:
        theta = 1.0*len(rank_list[0])/2
    scores = {}.fromkeys(rank_list[0])
    N = len(scores)
    item_rankings = item_ranks(rank_list)
    for k in scores:
        borda = sum([N - x for x in item_rankings[k]])
        ss = sum([i <= theta for i in item_rankings[k]])
        scores[k] = borda*ss
    return convert_to_ranks(scores)


def exponential_enhanced_borda(data,theta=None):
    print('exponential_enhanced_borda')
    """
    For exponential enhanced Borda, each item's Borda score is multiplied
    by its exponential weighting score and larger scores are assigned higher
    ranks.
    """
    
    rank_list = csv_to_dict(data)

    if theta is None:
        theta = 1.0*len(rank_list[0])/2
    scores = {}.fromkeys(rank_list[0])
    N = len(scores)
    item_rankings = item_ranks(rank_list)
    for k in scores:
        borda = sum([N - x for x in item_rankings[k]])
        expw = exp([-1.0*x/theta for x in item_rankings[k]]).sum()
        scores[k] = borda*expw
    return convert_to_ranks(scores)


def robust_aggregation(data):
    print('robust_aggregation')
    """
    Implements the robust rank aggregation scheme of Kolde, Laur, Adler,
    and Vilo in "Robust rank aggregation for gene list integration and
    meta-analysis", Bioinformatics 28(4) 2012.  Essentially compares
    order statistics of normalized ranks to a uniform distribution.
    """
    
    rank_list = csv_to_dict(data)

    def beta_calc(x):
        bp = np.zeros_like(x)
        n = len(x)
        for k in range(n):
            b = binom(n,x[k])
            for l in range(k,n):
                bp[k] += b.pmf(l+1)
        return bp
    scores = {}.fromkeys(rank_list[0])
    item_rankings = item_ranks(rank_list)
    N = len(scores)
    # sort and normalize the ranks, and then compute the item score
    for item in item_rankings:
        item_rankings[item] = np.sort([1.0*x/N for x in item_rankings[item]])
        # the 1.0 here is to make *large* scores correspond to better ranks
        scores[item] = 1.0 - min(beta_calc(item_rankings[item]))
    return convert_to_ranks(scores)


def round_robin(data):
    print('round_robin')
    """
    Round Robin aggregation.  Lists are given a random order.  The highest
    ranked item in List 1 is given rank 1 and then removed from consideration.
    The highest ranked item in List 2 is given rank 2, etc.  Continue until
    all ranks have been assigned.
    """
    
    rank_list = csv_to_dict(data)

    rr_ranks = {}.fromkeys(rank_list[0])
    N = len(rr_ranks)
    next_rank = 1
    next_list = 0
    # matrix of ranks
    rr_matrix = np.zeros((len(rr_ranks),len(rank_list)))
    items = list(rr_ranks.keys())
    # fill in the matrix
    for i in range(len(items)):
        for j in range(len(rank_list)):
            rr_matrix[i,j] = rank_list[j][items[i]]
    # shuffle the columns to randomize the list order
    rr_matrix = rr_matrix[:,permutation(rr_matrix.shape[1])]
    # start ranking
    while next_rank < N:
        # find the highest rank = lowest number in the row
        item_indx = np.argmin(rr_matrix[:,next_list])
        item_to_rank = items[item_indx]
        # rank the item and remove it from the itemlist and matrix
        rr_ranks[item_to_rank] = next_rank
        rr_matrix = np.delete(rr_matrix,item_indx,axis=0)
        items.remove(item_to_rank)
        next_rank += 1
        next_list = np.mod(next_list + 1,len(rank_list))
    # should only be one item left
    rr_ranks[items[0]] = N
    return rr_ranks


def footrule_aggregation(data):
    print('footrule_aggregation')
    """
    Computes aggregate rank by Spearman footrule and bipartite graph
    matching, from a list of ranks.  For each candiate (thing to be
    ranked) and each position (rank) we compute a matrix

        W(c,p) = sum(|tau_i(c) - p|)/S

    where the sum runs over all the experts doing the ranking.  S is a
    normalizer; if the number of ranks in the list is n, S is equal to
    0.5*n^2 for n even and 0.5*(n^2 - 1) for n odd.

    After constructing W(c,p), Munkres' algorithm is used for the linear
    assignment/bipartite graph matching problem.
    """
        
    ranklist = csv_to_dict(data)

    # lists are full so make an empty dictionary with the item keys
    items = ranklist[0].keys()
    # map these to matrix entries
    itemToIndex,indexToItem = item_mapping(items)
    print(itemToIndex)
    c = len(ranklist[0]) % 2
    scaling = 2.0/(len(ranklist[0])**2 - c)
    # thes are the positions p (each item will get a rank()
    p = range(1,len(items)+1)
    # compute the matrix
    W = np.zeros((len(items),len(items)))
    for r in ranklist:
        for item in items:
            taui = r[item]
            for j in range(0,len(p)):
                delta = abs(taui - p[j])
                # matrix indices
                W[itemToIndex[item],j] += delta
    W = scaling*W
    # solve the assignment problem
    path = linear_assignment(W)
    # construct the aggregate ranks
    aggRanks = {}
    for pair in path:
        aggRanks[indexToItem[pair[0]]] = p[pair[1]]
    return aggRanks



def locally_kemenize(data):
    print('locally_kemenize')
    """
    Performs a local kemenization of the ranks in aggranks and the list
    of expert rankings dictionaries in ranklist.  All rank lists must be full.
    The aggregate ranks can be obtained by any process - Borda, footrule,
    Markov chain, etc.  Returns the locally kemeny optimal aggregate
    ranks.

    A list of ranks is locally Kemeny optimal if you cannot obtain a
    lower Kendall tau distance by performing a single transposition
    of two adjacent ranks.
    """
    ranklist_ = csv_to_dict(data)

    aggRanks = aggregate_ranks(ranklist_,areScores=True,method='borda')
    
    ranklist = [convert_to_ranks(s) for s in ranklist_]
    
    # covert ranks to lists, in a consistent item ordering, to use
    # the kendall_tau_distance in metrics.py
    lkranks = {}
    items = list(aggranks.keys())
    sigma = [aggranks[i] for i in items]
    tau = []
    for r in ranklist_:
        tau.append([r[i] for i in items])
    # starting distance and distance of permuted list
    SKorig = 0
    # initial distance
    for t in tau:
        SKorig += kendall_tau_distance(sigma,t)
    # now try all the pair swaps
    for i in range(0,len(items)-1):
        SKperm = 0
        j = i + 1
        piprime = copy.copy(sigma)
        piprime[i],piprime[j] = piprime[j],piprime[i]
        for t in tau:
            SKperm += kendall_tau_distance(piprime,t)
        if SKperm < SKorig:
            sigma = piprime
            SKorig = SKperm
    # rebuild the locally kemenized rank dictionary
    for i in range(0,len(items)):
        lkranks[items[i]] = sigma[i]
    return lkranks


# ____________________________________________________________________________

from graphs import *

def MC1_getRanking(data, alpha, where):
    print('MC1_getRanking')
    ergodic_matrix = makeErgodic(MC1(data, where), alpha)
    ranking = list(fromScoresToRank(get_statDistr(ergodic_matrix)))
    
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


def MC2_getRanking(data, alpha, where):
    print('MC2_getRanking')
    ergodic_matrix = makeErgodic(MC2(data, where), alpha)
    ranking = list(fromScoresToRank(get_statDistr(ergodic_matrix)))
    
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


def MC3_getRanking(data, alpha, where):
    print('MC3_getRanking')
    ergodic_matrix = makeErgodic(MC3(data, where), alpha)
    ranking = list(fromScoresToRank(get_statDistr(ergodic_matrix)))
    
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


def MC4_getRanking(data, alpha, where):
    ergodic_matrix = makeErgodic(method1(data, where), alpha)
    ranking = list(fromScoresToRank(get_statDistr(ergodic_matrix)))
    
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


def MC5_getRanking(data, alpha, where):
    ergodic_matrix = makeErgodic(method2(data, where), alpha)
    ranking = list(fromScoresToRank(get_statDistr(ergodic_matrix)))
    
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

