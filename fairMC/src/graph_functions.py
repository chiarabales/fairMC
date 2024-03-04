import os
import pandas as pd
import numpy as np
import networkx as nx
from scipy.linalg import eig 
from graphs import *



        
def printGraphSums(G):
    
    # PRINT THE SUM OVER EXITING EDGES WEIGHTS (EXPECTED = 1)

    for source in G.nodes():
        sumsource = 0
        for target in G.nodes():
            if G.has_edge(source, target):
                sumsource += G[source][target]['weight']
        print(source, sumsource)
        
def printGraph(G):
    
    # PRINT THE GRAPHS WITH ALL THE WEIGHTS 
    
    for u, v, data in G.edges(data=True):
        print(f"Edge ({u}, {v}) has weight {data['weight']}")
        

def printGraphSumscheck(G, protected = 'W', unprotected = 'M'):
    
    # PRINT THE SUM OVER EXITING EDGES WEIGHTS FOR PROTECTED AND UNPROTECTED CATEGORIES (EXPECTED = 0.5)

    for source in G.nodes():
        protectedSum = 0
        unprotectedSum = 0

        for target in G.nodes():
            if G.has_edge(source, target):
                if protected in target:
                    protectedSum += G[source][target]['weight']
                elif 'M' in target:
                    unprotectedSum += G[source][target]['weight']
        print(f'source is {source} with protectedSum {protectedSum} and unprotectedSum {unprotectedSum}')
        
        
def makeFair(G, protected = 'W', unprotected = 'M'):
        
    fairG = nx.DiGraph()
    fairG.add_nodes_from(G.nodes())
    
    totalW = 0
    totalM = 0
    
    for source in G.nodes():  
        if unprotected in source:
            totalM += 1
        elif protected in source:
            totalW += 1
    print('totalM', totalM, 'totalW', totalW) 
    
    # prima moltiplicavamo per 0.5
    # ora calcolo

    factorW = totalW/(totalW + totalM)
    factorM = totalM/(totalW + totalM)
    
    print(factorW, factorM)

                
    for source in G.nodes():  
        sumToW = 0
        sumToM = 0
        sumTotal = 0
        
        for target in G.nodes():
            
            if G.has_edge(source, target):

                if protected in target:
                    sumToW += G[source][target]['weight']

                if unprotected in target:
                    sumToM += G[source][target]['weight']
                    
                sumTotal  += G[source][target]['weight']
                
#                 fairG.add_edge(source, target, weight = 0)



        
        ######todo is str() needed?
        for target in fairG.nodes():
            if sumToM != 0 and sumToW != 0:
                
                if (protected in target and G.has_edge(source, target)):
                    weight = G[source][target]['weight']*factorW/sumToW
                    fairG.add_edge(source, target, weight = weight)
                
                elif (unprotected in target and G.has_edge(source, target)):
                    weight = G[source][target]['weight']*factorM/sumToM
                    fairG.add_edge(source, target, weight = weight)

            if sumToW == 0 and sumToM != 0:
                if (protected in target):
                    weight = factorW/totalW
                    fairG.add_edge(source, target, weight = weight)
                
                elif (unprotected in target and G.has_edge(source, target)):
                    weight = G[source][target]['weight']*factorM/sumToM
                    fairG.add_edge(source, target, weight = weight)
                    
            if sumToW != 0 and sumToM == 0:
                
                if (protected in target and G.has_edge(source, target)):
                    weight = G[source][target]['weight']*factorW/sumToW
                    fairG.add_edge(source, target, weight = weight)
                
                elif (unprotected in target):
                    weight = factorM/totalM
                    fairG.add_edge(source, target, weight = weight)#             
                 
    return fairG


        
