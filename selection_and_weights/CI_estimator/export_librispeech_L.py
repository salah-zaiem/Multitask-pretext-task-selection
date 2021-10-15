import os 
import sys 
import pickle
import numpy as np
import os
from functools import partial
from tqdm import tqdm
from multiprocessing import Pool
import multiprocessing as mp
import pandas as pd 

csv_dir = sys.argv[1] #Dir where features are stored
feature = sys.argv[2] # Considered feature
words_dict_path = sys.argv[3] # Pkl with the words considered
outdir=sys.argv[4] # outdir


def rbf_value(x1,x2, sigma=0.05) : 
    diff = np.abs(x1-x2)**2
    return np.exp(-diff / (2*sigma)) 

#loading the pickle
with open(words_dict_path, 'rb') as f:
    words_dict = pickle.load(f)

number_words = 300

words = [x for x in words_dict if len(words_dict[x])>number_words]

#Efficient loading of the features
def load_vector(rank, vector_dicts, csv_dir, list_considered) :
    if rank in vector_dicts :
        return vector_dicts[rank],  vector_dicts
    else : 
        filename, start, end  = list_considered[rank].split("_")
        start, end = float(start), float(end)
        fname=  filename.split("/")[-1].split(".")[0] + ".csv"
        fname = os.path.join(csv_dir, fname)
        loaded = pd.read_pickle(fname)[feature]
        startframe = int(start * 100 )
        endframe = int(end*100)
        loaded = loaded[startframe:endframe]
        acts = np.mean(loaded)
        vector_dicts[rank] = acts
        return acts,  vector_dicts


def per_word_matrix(word) : 
    vector_dicts ={}
    N=number_words
    L_matrix = np.zeros((N,N))
    all_values = []
    list_considered = words_dict[word][0:number_words]
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
            if i!=j : 
                all_values.append(value)

    for i in range(N):
        for j in range(i,N): 
            L_matrix[i,j] = L_matrix[j,i]
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    np.save(os.path.join(outdir, "L_matrix_"+word+"_"+feature+".npy"), L_matrix)
part_func = per_word_matrix
parallel=False    
if parallel :
    v = mp.cpu_count()
    p = Pool(min(v, len(words)))
    print(f"working with {min(v,len(phones))} cpus")
    r = list(tqdm(p.imap(part_func, words), total=len(words)))
else : 
    for phone in tqdm(words) :
        part_func(phone)


