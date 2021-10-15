import os 
import sys 
from sklearn.preprocessing import normalize
import numpy as np
from multiprocessing import Pool
import multiprocessing as mp
import os
from tqdm import tqdm
from collections import Counter
from scipy.io.wavfile import read
import time
import pandas as pd
from functools import partial 
melfs_dir = sys.argv[1] # Directory with the mel spectrograms
gemap_dir = sys.argv[2] # Directory where the features extracted are stored
gemap_value = sys.argv[3] #Feature considered
outdir=sys.argv[4]
if not os.path.exists(outdir): 
    os.makedirs(outdir)
results_file= open(os.path.join(outdir,"results_per_speaker_"+gemap_value+".txt") , "w")

#Defining the RBF kernel
def rbf_value(x1,x2, sigma=0.05) : 
    diff = np.abs(x1-x2)**2
    to_be_returned = np.exp(-diff / (2*sigma)) 
    return to_be_returned
melfs_files = os.listdir(melfs_dir)
melfs_paths = [os.path.join(melfs_dir, x) for x in melfs_files]
speakers = list(np.unique([x.split("_")[1] for x in melfs_files]))
print(speakers)
csv_dir = gemap_dir
feature = gemap_value
#Efficient loader for the considered feature
def load_vector(rank, vector_dicts, csv_dir, list_considered) :
    if rank in vector_dicts :
        return vector_dicts[rank],  vector_dicts
    else : 
        filename= list_considered[rank]
        fname=  filename.split("/")[-1].split(".")[0] + ".csv"
        fname = os.path.join(csv_dir, fname)
        
        loaded = pd.read_pickle(fname)
        features = (loaded.columns)
        loaded = loaded[feature]
        acts = np.mean(loaded)
        vector_dicts[rank] = acts
        return acts,  vector_dicts


def per_speaker_matrix(speaker, paths) : 
    print(f"started speaker :{speaker}")
    melfs_paths =[paths[x] for x in range(len(paths)) if melfs_files[x].split("_")[1]==speaker]
    N=len(melfs_paths)
    L_matrix = np.zeros((N,N))
    all_values = []
    vector_dicts ={}
    list_considered =  melfs_paths
    for i in tqdm(range(N)):
        svacts1,  vector_dicts = load_vector(i,
                                                     vector_dicts,
                                                               csv_dir,
                                                               list_considered)
        for j in range(i+1): 
            svacts2, vector_dicts = load_vector(j,
                                               vector_dicts,
                                                               csv_dir,
                                                                   list_considered)
            value = rbf_value(svacts1, svacts2)


            L_matrix[i,j]=value

    for i in range(N):
        for j in range(i,N): 
            L_matrix[i,j] = L_matrix[j,i]
    np.save(os.path.join(outdir, "L_matrix_" + speaker + "_"+gemap_value+".npy"), L_matrix)
    return 1
v = mp.cpu_count()
parallel=False
Ns = []
for speaker in speakers : 
    M =len([melfs_paths[x] for x in range(len(melfs_paths)) if  melfs_files[x].split("_")[0]==speaker])
    Ns.append(M)
part_func = partial(per_speaker_matrix,paths = melfs_paths) 
if parallel : 
    cpus_needed = min(len(speakers),v)
    print(f"using {cpus_needed} cpus")
    p = Pool(cpus_needed)
    lens = len(speakers)
    job_done = list(tqdm(p.imap(part_func,speakers), total=lens))
else : 
    results_here =[]
    for speaker in tqdm(speakers): 

        results_here.append(part_func(speaker))


