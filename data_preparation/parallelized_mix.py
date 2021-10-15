import numpy as np 
import pandas as pd
from scipy.io.wavfile import read, write
from tqdm import tqdm
import os 
import sys
import librosa
from sphfile import SPHFile
import csv
import multiprocessing as mp
from multiprocessing import Pool
print(f" numbre of cpus : {mp.cpu_count()}")


dir1=sys.argv[1]
dir2=sys.argv[2]
dir3 = sys.argv[3]
if not os.path.exists(dir3):
    os.makedirs(dir3)
elements = os.listdir(dir1)
def f(i):
    element = elements[i]
    csv1= pd.read_pickle(os.path.join(dir1,element))
    csv2= pd.read_pickle(os.path.join(dir2,element))
    result = pd.concat([csv1,csv2], axis=1)
    result.to_pickle(os.path.join(dir3, element))
listing2= os.listdir(dir2)
p = Pool(mp.cpu_count())
N= len(elements)
#for i in range(N) : 
#    f(i)
parallel=True
if parallel :
    r = list(tqdm(p.imap(f, range( N)), total=N))
else : 
    for i in range(N): 
        f(i)
