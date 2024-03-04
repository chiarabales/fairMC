#functions

from metrics import *
from collections import Counter
import math

def GrBinaryIPF(ranking, f_values, protected_attribute):
    n = len(ranking)
    bot_dict = {}  # Dictionary to store bot values for each item
    final_ranking = []  # List to store the final ranking

    # Step 1: Partition the sorted list into two sub-lists L1 and L2
    L1 = [item for item in ranking if item[0] == protected_attribute]
    L2 = [item for item in ranking if item[0] != protected_attribute]

    # Step 3: Compute bot(v) for each item v in V
    for item in ranking:
        class_id = item[0]
        if class_id == protected_attribute:
            bot = math.ceil((L1.index(item) +1) / f_values[class_id])  # i+1 because the first index is 1
        if class_id != protected_attribute:
            bot = math.ceil((L2.index(item) +1) / f_values[class_id])  # i+1 because the first index is 1
        
        bot_dict[item] = bot

    # Step 4: Loop from 1 to n
    for i in range(1, n + 1):
        # Step 5: Get the current top items in L1 and L2
        u1 = L1[0] if L1 else None
        u2 = L2[0] if L2 else None

        # Step 6: Check if bot(u1) = i or bot(u2) = i
        if u1 and bot_dict[u1] == i:
            v = u1
            L1.remove(v)
        elif u2 and bot_dict[u2] == i:
            v = u2
            L2.remove(v)
        else:
            # Step 9: Choose the higher-ranked item
            if u1 and u2:
                index_u1 = ranking.index(u1)
                index_u2 = ranking.index(u2)
                if index_u1 < index_u2:
                    v = u1
                    L1.remove(v)
                else:
                    v = u2
                    L2.remove(v)
            elif u1:
                v = u1
                L1.remove(v)
            elif u2:
                v = u2
                L2.remove(v)

        # Step 11: Append v to the final ranking
        final_ranking.append(v)

    return final_ranking
    
def bestRankAggregated(data):
    
    result_dict = {}
    
    df_for_pick_a_perm = pd.DataFrame(columns=data.columns)
    
    # questa dovrebbe essere una funzione, controllare
    for col in data.columns:
        sorted_indices = data[col].sort_values(ascending=True).index
        df_for_pick_a_perm[col] = sorted_indices
        
    

    df_to_pick_best = pd.DataFrame()
    co = Counter([x[0] for x in data.index])

    total_elements = data.shape[0]

    fractions = {group: count / total_elements for group, count in co.items()}

    for i,_i in enumerate(df_for_pick_a_perm.columns):
        df_to_pick_best[_i] = GrBinaryIPF(list(df_for_pick_a_perm.iloc[:, i]), fractions, 'W')
    


    for column in df_to_pick_best.columns:
        column_values = df_to_pick_best[column]
        val_df = pd.DataFrame({column: range(1, len(column_values) + 1)}, index=column_values)
        result_dict[f'C {column}'] = val_df


    # Merge all the DataFrames into one
    merged_df_to_pick_best = pd.concat(result_dict, axis=1)
    merged_df_to_pick_best = merged_df_to_pick_best.loc[data.index]

    
    best_picked = []
    for i in range(merged_df_to_pick_best.shape[1]):
        dist = 0
        for j in range(data.shape[1]):
            dist += kendall_tau_distance_normalized(merged_df_to_pick_best.iloc[:, i], data.iloc[:, j])
        dist /= len(data)
        best_picked.append(dist)
    result = merged_df_to_pick_best.iloc[:, np.argmin(best_picked) ]
    return result.to_dict()

