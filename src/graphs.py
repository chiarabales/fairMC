import os
import pandas as pd
import numpy as np
import networkx as nx
from scipy.linalg import eig 

def method1(data, where):
    path_data = '../data/' + where + '/'
    df = pd.read_csv(path_data + data + '.csv', index_col = 0)

    S = df.shape[1]
    col = df.shape[0]

    # init graph
    # Getting the item names (nodes) from the DataFrame
    item_nodes = df.index.tolist()
    # Creating an empty NetworkX graph
    G = nx.DiGraph()
    # Adding nodes to the graph
    G.add_nodes_from(item_nodes)
    
    #create a set where each item is the sorted values of a colum
    sorted_columns = {}
    for column in df.columns:
        sorted_columns[column] = df[column].sort_values().index
    
    #create edges or increment weight for consecutive items
    for column, order in sorted_columns.items():
        mylist = order
        for i in range(len(mylist)-1):
            source = mylist[i + 1]
            target = mylist[i]

            # Check if the edge already exists and get the existing weight
            if G.has_edge(source, target):
                existing_weight = G[source][target]['weight']
            else:
                existing_weight = 0

            # Add the edge with the updated weight (existing + 1)
            G.add_edge(source, target, weight=(existing_weight + 1))
                    
    for source in G.nodes():
        W = 1
        for target in G.nodes():
            if G.has_edge(source, target):
                W -= G[source][target]['weight']
            else:
                W=W
        G.add_edge(source, source, weight=W)
        
    Gnorm = normalize_graph(G)
    return Gnorm


def method2(data, where):
    
    # the method connect each element will all the elements that are below it
    
    path_data = '../data/' + where + '/'
    df = pd.read_csv(path_data + data + '.csv', index_col = 0)
    
    # Getting the item names (nodes) from the DataFrame
    # this are the elements to be ranked
    
    item_nodes = df.index.tolist()
    
    S = df.shape[0]
    col = df.shape[1]
    
    # Creating an empty NetworkX graph and add the ranked items as nodes
    G = nx.DiGraph()
    G.add_nodes_from(item_nodes)
    
    #create a set where each item is the sorted values of a colum
    
    sorted_columns = {}
    for column in df.columns:
        sorted_columns[column] = df[column].sort_values().index

    for column, order in sorted_columns.items():
        mylist = order
        for i in range(len(mylist)-1):
            sources = mylist[i + 1:] # here we create connections with all below!
            target = mylist[i]

            # Loop through targets and add edges
            for source in sources:
                # Check if the edge already exists and get the existing weight
                if G.has_edge(source, target):
                    existing_weight = G[source][target]['weight']
                else:
                    existing_weight = 0

                # Add the edge with the updated weight (existing + 1)
                G.add_edge(source, target, weight=(existing_weight + 1)/(S*col))
    
    
    for source in G.nodes():
        W = 1
        for target in G.nodes():
            if G.has_edge(source, target):
                W -= G[source][target]['weight']
            else:
                W=W
        G.add_edge(source, source, weight=W)
    Gnorm = normalize_graph(G)
    return Gnorm

# ___________________________________________________________________________________

def MC1(data, where):
    
    path_data = '../data/' + where + '/'
    df = pd.read_csv(path_data + data + '.csv', index_col = 0)
    
    S = df.shape[0]
    
    # Getting the item names (nodes) from the DataFrame
    # this are the elements to be ranked
    
    item_nodes = df.index.tolist()
    
    # Creating an empty NetworkX graph and add the ranked items as nodes
    G = nx.DiGraph()
    G.add_nodes_from(item_nodes)
    
    #create a set where each item is the sorted values of a colum

    for i in item_nodes:
        for j in item_nodes:
            
            for column in df.columns:
                if (df.loc[i][column] > df.loc[j][column] and i != j):
                    source = i # here we create connections with all below!
                    target = j
                    G.add_edge(source, target, weight=1/S)
                    break
                    
    for source in G.nodes():
        W = 1
        for target in G.nodes():
            if G.has_edge(source, target):
                W -= G[source][target]['weight']
            else:
                W=W
        G.add_edge(source, source, weight=W)
    Gnorm = normalize_graph(G)
    return Gnorm

def MC2(data, where):
    
    path_data = '../data/' + where + '/'
    df = pd.read_csv(path_data + data + '.csv', index_col = 0)
    
    S = df.shape[0]
    col = df.shape[1]
    
    # Getting the item names (nodes) from the DataFrame
    # this are the elements to be ranked
    
    item_nodes = df.index.tolist()
    
    # Creating an empty NetworkX graph and add the ranked items as nodes
    G = nx.DiGraph()
    G.add_nodes_from(item_nodes)
    
    
    #create a set where each item is the sorted values of a colum

    for i in item_nodes:
        for j in item_nodes:
            counter = 0
            for column in df.columns:
                if (df.loc[i][column] > df.loc[j][column] and i != j):
                    source = i # here we create connections with all below!
                    target = j
                    counter += 1
               
                    if counter > col/2:
                        G.add_edge(source, target, weight=1/S)
                        break
                    
    for source in G.nodes():
        W = 1
        for target in G.nodes():
            if G.has_edge(source, target):
                W -= G[source][target]['weight']
            else:
                W=W
        G.add_edge(source, source, weight=W)
    Gnorm = normalize_graph(G)
    return Gnorm

def MC3(data, where):
    
    path_data = '../data/' + where + '/'
    df = pd.read_csv(path_data + data + '.csv', index_col = 0)
    
    # Getting the item names (nodes) from the DataFrame
    # this are the elements to be ranked
    
    S = df.shape[0]
    col = df.shape[1]
    item_nodes = df.index.tolist()
    
    # Creating an empty NetworkX graph and add the ranked items as nodes
    G = nx.DiGraph()
    G.add_nodes_from(item_nodes)
    
    
    #create a set where each item is the sorted values of a colum

    for source in item_nodes:
        for target in item_nodes:             
            counter = 0
            for column in df.columns:
                if (df.loc[source][column] > df.loc[target][column] and source != target):
                    counter += 1
            G.add_edge(source, target, weight=counter/(S*col))
                    
    for source in G.nodes():
        W = 1
        for target in G.nodes():
            if G.has_edge(source, target):
                W -= G[source][target]['weight']
            else:
                W=W
        G.add_edge(source, source, weight=W)
    Gnorm = normalize_graph(G)
    return Gnorm



# ___________________________________________________________________________________

# functions on graphs

def normalize_graph(G):

    #Calculate the sum of weights for each source node
    source_sums = {}
    for source, target, weight in G.edges(data=True):
        source_sums[source] = source_sums.get(source, 0) + weight['weight']

    #Normalize the weights
    for source, target, weight in G.edges(data=True):
        if source_sums[source] != 0:
            weight['weight'] /= source_sums[source]
            
    return G

def makeErgodic(G1, alpha):
    
    S = len(G1.nodes())
    G = nx.DiGraph()
    G.add_nodes_from(G1.nodes())
    
    for source in G1.nodes():
        for target in G1.nodes():
            if G1.has_edge(source, target):
                existing_weight = G1[source][target]['weight']
            else:
                existing_weight = 0
                
            G.add_edge(source, target, weight=alpha*existing_weight + (1-alpha)/S)
            
    return G


# ___________________________________________________________________________________

def get_statDistr(G):
    adjacency_matrix = nx.adjacency_matrix(G).toarray()
    transition_matrix = adjacency_matrix / adjacency_matrix.sum(axis=1)
    
    _, left_eigenvectors = eig(transition_matrix, left=True, right=False)
    stationary_vector = left_eigenvectors[:, 0].real
    
    stationary_distribution = stationary_vector / stationary_vector.sum()
    return stationary_distribution

def check_ifProbDist(array):
    is_prob = True
    for item in array:
        if item < 0:
            is_prob = False
            break
    return is_prob

def fromScoresToRank(scores, descending = True):
    
    if check_ifProbDist(scores) == False:
        print('it is not a probability distribution')
        return np.arange([0 for i in range(len(scores))])

        
    else:
        # from the probability function, get the rank
        sorted_indices = np.argsort(scores)

        if descending== True:
            sorted_indices = sorted_indices[::-1]  # Reverse the order for descending rank
        rankings = np.empty_like(sorted_indices)
        rankings[sorted_indices] = np.arange(len(scores)) + 1
    
    return rankings