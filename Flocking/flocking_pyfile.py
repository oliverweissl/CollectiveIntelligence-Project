import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
import pandas as pd

df = pd.read_csv("Experiments/A0.75_C0.40_S0.40_W0.50.csv")
print(df)

in_flocks = []
angles = []
clusters_num_tt = []
indiv_tt = []

def find_in_cluster(cl, idx):
    for k, cluster in cl.items():
        if idx in cluster:
            return k
    return None

def neighbours(indiv, i, curr_cluster, counter):
    ind = indiv[i]
    reached_ind = []
    for i_2 in range(len(indiv)):
        if indiv[i_2][5]:
            continue

        if i_2 != i and \
                np.deg2rad(np.abs(ind[2] - indiv[i_2][2])) < 0.5 and \
                np.linalg.norm(ind[:2] - indiv[i_2][:2]) <= 25:
            indiv[i_2][5] = curr_cluster
            if counter == 128:
                print(i, ind, i_2, indiv[i_2])
            reached_ind.append(i_2)
    for j in reached_ind:
        neighbours(indiv, j, curr_cluster, counter)    
    curr_cluster += 1
    return indiv, curr_cluster

counter = 0

for frame in tqdm(range(max(df["frame"]))):
    curr_cluster = 1
    in_flocks.append(df["image_index"][(df["frame"]==frame) & (df["image_index"]==1)].count())
    #print(in_flocks)
    indiv = df[["x","y","angle", "image_index", "id"]][df["frame"]==frame]#.to_numpy()#.reshape(-1, 1)
    indiv["cluster"] = None
    indiv = indiv.to_numpy()
    
    for i in range(len(indiv)):
        if indiv[i][5]:
            continue
        indiv[i][5] = curr_cluster
        indiv, curr_cluster = neighbours(indiv, i, curr_cluster, counter)
        #print(indiv[:, 4])
    clusters_num_tt.append(curr_cluster)
    counter += 1
    indiv_tt.append(indiv)
print(clusters_num_tt)

# indiv_tt is a num_frames*num_individuals*[x, y, angle, image_index, id, cluster_index] shaped list
# Use it to create a line chart of all the clusters of individuals through time, where x-axis is time and y-axis is the average angle of a given cluster
# Line chart should have as many lines at a given timestep as there are clusters of size more than 2
# if 2 flocks combine, then the lines on the chart combine into one line
# Solution to the problem should keep track of cluster index. Use individual id's to your advantage.
                
#np.arccos(np.dot(np.average(vec,axis = 0), self.move))< 0.5
