import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
import pandas as pd
import pickle 

df = pd.read_csv("Experiments/A0.80_C0.20_S0.20_W0.30.csv")
df["cluster"] = None

def find_in_cluster(cl, idx):
    for k, cluster in cl.items():
        if idx in cluster:
            return k
    return None

def neighbours(indiv, i, curr_cluster, frame):
    ind = indiv[i]
    reached_ind = []
    for i_2 in range(len(indiv)):
        if indiv[i_2][5] is not None:
            continue
        if i_2 != i and np.deg2rad(np.abs(ind[2] - indiv[i_2][2])) < 0.5 and np.linalg.norm(ind[:2] - indiv[i_2][:2]) <= 25:
            indiv[i_2][5] = curr_cluster
            reached_ind.append(i_2)
            neighbours(indiv, i_2, curr_cluster, frame)
            """
            if frame == 128:
                print(i, ind, i_2, indiv[i_2])
            """

   # for j in reached_ind:
        
    
    return indiv, curr_cluster


indiv_tt = []
clusters_num_tt = []

for frame in tqdm(range(max(df["frame"]))):
    curr_cluster = 0
    indiv = df[["x","y","angle", "image_index", "id","cluster"]][df["frame"]==frame].to_numpy()
    for i,ind in enumerate(indiv):
        if ind[5]:
            continue
        ind[5] = curr_cluster
        indiv, curr_cluster = neighbours(indiv, i, ind[5], frame)
        curr_cluster += 1

    clusters_num_tt.append(curr_cluster)
    indiv_tt.append(indiv)
print(clusters_num_tt)

pickle.dump(clusters_num_tt, open("clusters_num_tt_bad.pickle", "wb"))
# indiv_tt is a num_frames*num_individuals*[x, y, angle, image_index, id, cluster_index] shaped list
# Use it to create a line chart of all the clusters of individuals through time, where x-axis is time and y-axis is the average angle of a given cluster
# Line chart should have as many lines at a given timestep as there are clusters of size more than 2
# if 2 flocks combine, then the lines on the chart combine into one line
# Solution to the problem should keep track of cluster index. Use individual id's to your advantage.
                
#np.arccos(np.dot(np.average(vec,axis = 0), self.move))< 0.5
