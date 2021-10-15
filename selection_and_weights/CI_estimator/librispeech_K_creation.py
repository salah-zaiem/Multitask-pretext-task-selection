import os 
import sys 
import pickle
import numpy as np
import os
from functools import partial
from tqdm import tqdm
from multiprocessing import Pool
import multiprocessing as mp
import cca_core
from gaussian_relative_nonUniform_downsample_uniform import downsample


melfs_dir = sys.argv[1]
words_dict_path = sys.argv[2]
outdir=sys.argv[3]

#The word dict path contains a list with words position ( from word alignment ) in speech
# For ASR words are used as the downstream classes
with open(words_dict_path, 'rb') as f:
    words_dict = pickle.load(f)

number_words = 300

words = [x for x in words_dict if len(words_dict[x])>number_words]
# Now building the K matrix : 
melfs_files = os.listdir(melfs_dir)
melfs_paths = [os.path.join(melfs_dir, x) for x in melfs_files]

gaussian_downsampling = partial(downsample, n_samples= 10, std_ratio=0.07,
                                std_slope=0.1)

#Efficient loading of the mels needed, cutting is involved to get the spectrograms corresponding to the considered word
def load_vector(rank, vector_dicts, norm_dicts, melfs_dir, list_considered) :
    if rank in vector_dicts :
        return vector_dicts[rank], norm_dicts[rank], vector_dicts, norm_dicts
    else : 
        filename, start, end  = list_considered[rank].split("_")
        start, end = float(start), float(end)
        fname=  filename.split("/")[-1].split(".")[0] + ".npy"
        loaded= np.load(os.path.join(melfs_dir, fname))
        startframe = int(start * 100 )
        endframe = int(end*100)
        loaded = loaded[startframe:endframe]
        acts = gaussian_downsampling(loaded)
        norm = np.linalg.norm(acts)
        norm_dicts[rank] = norm
        vector_dicts[rank] = acts
        return acts, norm, vector_dicts, norm_dicts
#The per_word_matrix corresponds to the function for the other tasks. 
def per_word_matrix(word, paths) : 
    vector_dicts ={}
    norm_dicts = {}
    N=number_words
    K_matrix = np.zeros((N,N))
    all_values = []
    list_considered = words_dict[word][0:number_words]
    for i in tqdm(range(N)):
        svacts1, norm1, vector_dicts, norm_dicts = load_vector(i,
                                                               vector_dicts,norm_dicts,
                                                               melfs_dir,
                                                               list_considered)
        for j in range(i+1): 
            svacts2, norm2, vector_dicts, norm_dicts = load_vector(j,
                                                               vector_dicts,norm_dicts,
                                                               melfs_dir,
                                                                   list_considered)
            scalar = np.trace(svacts1.T @ svacts2)
            value = scalar / (norm1*norm2)

            K_matrix[i,j]=value
            if i!=j : 
                all_values.append(value)

    for i in range(N):
        for j in range(i,N): 
            K_matrix[i,j] = K_matrix[j,i]
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    np.save(os.path.join(outdir, "K_matrix_"+word+".npy"), K_matrix)
part_func = partial(per_word_matrix,paths = melfs_paths) 
parallel=False    #parallel or sequential implementation
if parallel :
    v = mp.cpu_count()
    p = Pool(min(v, len(words)))
    print(f"working with {min(v,len(phones))} cpus")
    r = list(tqdm(p.imap(part_func, words), total=len(words)))
else : 
    for phone in tqdm(words) :
        part_func(phone)


