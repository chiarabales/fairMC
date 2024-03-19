import pandas as pd
import numpy as np
import click
import utils


def fairness_loss(data, k):
    
    elements = data.index

    fairness_gained = []
    total_protected = np.sum(['W' in item for item in elements])
    total_NONprotected = np.sum(['W' not in item for item in elements])

    
    for method in data.columns:
        protected = 0
        non_protected = 0
    
        l = np.asarray(data[data[method] <= k].index)
        for j in range(k):
            if 'W' in l[j]:
                protected += 1
            else:
                non_protected += 1
        fairness_gained.append(protected/total_protected*100 - total_protected/total_NONprotected*100)
    return fairness_gained


def group_exposure(data):

    elements = data.index
    exposure_method = []
    

    total_protected = np.sum(['W' in item for item in elements])
    total_NONprotected = np.sum(['W' not in item for item in elements])

    for method in data.columns:
        protected = 0
        non_protected = 0
        
        sorted_indices = data[method].sort_values().index
        new_vector = pd.Series(range(1, len(sorted_indices) + 1), index=sorted_indices)
        


        for j, name in enumerate(new_vector.index):
            if 'W' in name:
                protected += 1/(np.log2(j+1)+1)
            else:
                non_protected += 1/(np.log2(j+1)+1)

        exposure_method.append(non_protected/total_NONprotected  - protected/total_protected)
    
    return exposure_method



# _______________________________________________________________________________________________________

# KACHEL et al. - fairer together

# _______________________________________________________________________________________________________


def precedence_matrix_disagreement(baseranks):
    """
    :param baseranks: num_rankers x num_items
    :return: precedence matrix of disagreeing pair weights. Index [i,j] shows # disagreements with i over j
    """
    num_rankers, num_items = baseranks.shape


    weight = np.zeros((num_items, num_items))

    pwin_cand = np.unique(baseranks[0]).tolist()
    plose_cand = np.unique(baseranks[0]).tolist()
    combos = [(i, j) for i in pwin_cand for j in plose_cand]
    for combo in combos:
        i = combo[0]
        j = combo[1]
        h_ij = 0 #prefer i to j
        h_ji = 0 #prefer j to i
        for r in range(num_rankers):
            if np.argwhere(baseranks[r] == i)[0][0] > np.argwhere(baseranks[r] == j)[0][0]:
                h_ij += 1
            else:
                h_ji += 1

        weight[i, j] = h_ij
        weight[j, i] = h_ji
        np.fill_diagonal(weight, 0)
    return weight

def precedence_matrix_agreement(baseranks):
    """
    :param baseranks: num_rankers x num_items
    :return: precedence matrix of disagreeing pair weights. Index [i,j] shows # agreements with i over j
    """
    num_rankers, num_items = baseranks.shape


    weight = np.zeros((num_items, num_items))

    pwin_cand = np.unique(baseranks[0]).tolist()
    plose_cand = np.unique(baseranks[0]).tolist()
    combos = [(i, j) for i in pwin_cand for j in plose_cand]
    for combo in combos:
        i = combo[0]
        j = combo[1]
        h_ij = 0 #prefer i to j
        h_ji = 0 #prefer j to i
        for r in range(num_rankers):
            if np.argwhere(baseranks[r] == i)[0][0] < np.argwhere(baseranks[r] == j)[0][0]:
                h_ij += 1
            else:
                h_ji += 1

        weight[i, j] = h_ij
        weight[j, i] = h_ji
        np.fill_diagonal(weight, 0)
    return weight

def calc_consensus_accuracy(base_ranks, consensus):
    agree_count = 0
    n_voters, n_items = np.shape(base_ranks)
    print(n_voters, n_items)
    precedence_mat = precedence_matrix_agreement(base_ranks)
    positions = len(consensus)
    for pos in range(positions):
        won = consensus[pos]
        lost = consensus[pos + 1: positions]
        for x in lost:
            agree_count += precedence_mat[won, x]

    print("agree count", agree_count)
    print("sum precedence_mat", np.sum(precedence_mat))
    result = agree_count/np.sum(precedence_mat)
    return result


def calcConsensusAccuracy(_data_results, _data):
    rankers, elements = _data.shape
    base_ranks = _data.to_numpy().T -1
    consACC = []
    for col in _data_results.columns:
        consensus = np.asarray(_data_results[col]) -1
        consACC.append(calc_consensus_accuracy(base_ranks, consensus))
    return consACC


# _______________________________________________________________________________________________________

#                                                WEI ET AL.

# _______________________________________________________________________________________________________

def R_parOne(ranking, protected = 'W'):
            
    N = ranking.shape[0]
    
    numberProt = 0
    numberUnProt = 0
    
    indexProt = set()
    indexUnProt = set()
    
    for index_name in ranking.index:
        if 'W' in index_name:
            numberProt += 1
            indexProt.add(index_name)
        else: 
            numberUnProt += 1
            indexUnProt.add(index_name)
            
    Rp = 0
    for prot in indexProt:
        for unprot in indexUnProt:
            if (ranking.loc[prot] <= ranking.loc[unprot]):
                Rp += 1 
            else:
                Rp -= 1
    return np.abs(Rp/(numberUnProt*numberProt))

def R_par(data, protected = 'W'):
    
    R_p = []
    for ranking in data.columns:
         R_p.append(R_parOne(data[ranking], protected))
            
    return R_p


# _______________________________________________________________________________________________________


def kendall_tau_distance_normalized(rank1, rank2):
    if len(rank1) != len(rank2):
        raise ValueError("Both rankings must have the same length")

    n = len(rank1)
    distance = 0

    for i in range(n):
        for j in range(i + 1, n):
            if (rank1[i] < rank1[j] and rank2[i] > rank2[j]) or (rank1[i] > rank1[j] and rank2[i] < rank2[j]):
                distance += 1

    max_distance = (n * (n - 1)) / 2
    normalized_distance = distance / max_distance

    return normalized_distance

def kemeny_distance_norm(aggregated_ranking, df):
    '''Take as input a target ranking, as a list, and compute the kemeny distance between the target ranking and the rankings stored in a dataframe'''
    
    # aggregated_ranking: list of sorted elements
    # df: dataframe of the original rankings from which we get the aggregated_ranking 
    
    # ----------------------------------------------------------------
    # --------------------- remarks : --------------------------------
    # ----------------------------------------------------------------
    
    # it is already normalized with respect to the number of ranking
    
    kendall_tau_distances = {}
    kem_dist = 0
    for column in df.columns:
        ranking = df[column].tolist()
        dist = kendall_tau_distance_normalized(aggregated_ranking, ranking)
        kem_dist += dist
    
    
    return kem_dist/(len(df.columns))

def kemeny_distances(aggregated_results, df):
    distances = []
    
    aggregated_results = aggregated_results.sort_index()
    df = df.sort_index()
    
    for method in aggregated_results.columns:
        aggregated_ranking = aggregated_results[method].tolist()
        distances.append(kemeny_distance_norm(aggregated_ranking, df))
        
    return distances
        
# _______________________________________________________________________________________________________

# TOP K PARITY

def top_k_parity(ranking, protected = 'W'):
    
    # works only for binary 
    # protected value is W
    
    rankW = []
    for i in ranking.index:
        if protected in i:
            rankW.append(i)

    numW = len(rankW)
    propW = numW/ranking.shape[0]
    
    candW = 0
    topkParity = []
    for k in range(1, ranking.shape[0]):
        for item in rankW:
            if ranking.loc[item] <= k:
                candW += 1
        if np.abs(candW/k - propW)*100/propW <= 10:
            topkParity.append(True)
        else:
            topkParity.append(False)
    return topkParity


def topKpar(data, protected = 'W'):
    
    topK = []
    for ranking in data.columns:
         topK.append(top_k_parity(data[ranking], protected))
            
    return topK
    

#_______________________________________________________________________________________________________

# using only one rankings

def compute_protected(ranking, k):
    
    prot_atk = 0
    prot = 0
    for i in list(ranking):
        if i[0] == 'W':
            if list(ranking).index(i) <= k:
                prot_atk += 1
            prot += 1
            
    return prot_atk, prot

def rND_ranking(ranking, k): 
    
    # REVERSED

    prot_atk, prot = compute_protected(ranking, k)
    
    return abs(prot_atk/k-prot/len(ranking))


def rRD_ranking(ranking, k):

    # REVERSED
    
    prot_atk, prot = compute_protected(ranking, k)
    
    input_ratio = prot/(len(ranking)-prot)
    unprot_atk = k-prot_atk
    
    if unpro_k==0: # manually set the case of denominator equals zero
        ratio = 0
    else:
        ratio = prot_atk/unprot_atk

    return abs(min(input_ratio,ratio)-input_ratio)


def KL_ranking(ranking, k):
    
    # REVERSED
    
    # defined in Yang and Stoyanovich
    # confronta la probabilità di elementi prtetti a k rispetto a quanti elementi protetti ci sono..
    # take as input a column of the dataframe (already transformed), i.e., r = dic_Reversed['ranker 1']
    # k is the maximum element considered in the ranking 
    
    prot_atk, prot = compute_protected(ranking, k)

    par_prob = prot_atk/k
    tot_prob = prot/len(ranking)
            
    if  par_prob in [0,1]:
        par_prob = 0.0001
    if  tot_prob in [0,1]:
        tot_prob = 0.0001

    return par_prob*math.log(par_prob/tot_prob,2)+(1-par_prob)*math.log((1-par_prob)/(1-tot_prob),2)

#_______________________________________________________________________________________________________

# aggregating ...


def KL(data, k):
    
    conv_data = utils.convert_dictionary(data)
    kls = []
    for method in conv_data.columns:
        kls.append(KL_ranking(conv_data[method], k))
        
    return kls

def rND(data, k):
    
    conv_data = utils.convert_dictionary(data)
    rnds = []
    for method in conv_data.columns:
        rnds.append(rND_ranking(conv_data[method], k))
        
    return rnds

def rRD(data, k):
    
    conv_data = utils.convert_dictionary(data)
    rrds = []
    for method in conv_data.columns:
        rrds.append(rRD_ranking(conv_data[method], k))
        
    return rrds







def metric_Chakraborty():
    return False

def metric_wei():
    # Proportionate Fair or p-fair ranking
    return False

def Pairwise_statistical_parity():
     # in Kuhlman (defined also in wei)
    return False

def Geyik():
    # defined in wei
    return False

def Topk_Parity():
    return False

def Rank_Equality_Error():
    return False

