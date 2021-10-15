import os 
import sys 

import numpy as np
import os
from functools import partial
from tqdm import tqdm
from multiprocessing import Pool
import multiprocessing as mp
from collections import Counter
import cca_core
melfs_dir = sys.argv[1] #Directory with files containing for every file the mel spectrogram of the corresponding speech sample.
outdir=sys.argv[2]
# Now building the K matrix : 
melfs_files = os.listdir(melfs_dir)
melfs_paths = [os.path.join(melfs_dir, x) for x in melfs_files]
#In this file, we consider that the downstream class is the readable within the file name 
# if x is the filename x.split("_")[1] should be the downstream class id
# you can change this according to your use.
speakers = list(np.unique([x.split("_")[1] for x in melfs_files]))
#print(speakers)
from gaussian_relative_nonUniform_downsample_uniform import downsample
#Defining the gaussian downsampling parameters
gaussian_downsampling = partial(downsample, n_samples= 30, std_ratio=0.07,
                                std_slope=0.1)


def svd_transform(melfs1, size=20) :
    cacts1 = melfs1 - np.mean(melfs1, axis=1, keepdims=True)

    U1, s1, V1 = np.linalg.svd(cacts1, full_matrices=False)

    svacts1 = np.dot(s1[:size]*np.eye(size), V1[:size])
   
    return svacts1

def per_speaker_matrix(speaker, paths) : 
    melfs_paths =[paths[x] for x in range(len(paths)) if melfs_files[x].split("_")[1]==speaker]
    N=len(melfs_paths)
    K_matrix = np.zeros((N,N))
    all_values = []
    for i in (range(N)):
        loaded_i = np.load(melfs_paths[i])
        # Mean subtract activations
        #svacts1= svd_transform(loaded_i)
        svacts1 = gaussian_downsampling(loaded_i)
        norm1 = np.linalg.norm(svacts1)
        # can also compute as svacts1 = np.dot(U1.T[:20], cacts1)
        for j in range(i+1): 
            loaded_j= np.load(melfs_paths[j])
            #svacts2 = svd_transform(loaded_j)
            svacts2=gaussian_downsampling(loaded_j)
            norm2 = np.linalg.norm(svacts2) 
            scalar = np.trace(svacts1.T @ svacts2)
            value = scalar / (norm1*norm2)

            #results =cca_core.get_cca_similarity(svacts1, svacts2,
            #                                      verbose=False)
            #value = np.mean(results["cca_coef1"])
            K_matrix[i,j]=value
            if i!=j : 
                all_values.append(value)

    for i in range(N):
        for j in range(i,N): 
            K_matrix[i,j] = K_matrix[j,i]
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    np.save(os.path.join(outdir, "K_matrix_"+speaker+".npy"), K_matrix)
part_func = partial(per_speaker_matrix,paths = melfs_paths) 
parallel=True    
if parallel :
    v = mp.cpu_count()
    p = Pool(min(v, len(speakers)))
    print(f"working with {min(v,len(speakers))} cpus")
    r = list(tqdm(p.imap(part_func, speakers), total=len(speakers)))
else : 
    for speaker in speakers :
        part_func(speaker)
#Start by stocking the melfs


