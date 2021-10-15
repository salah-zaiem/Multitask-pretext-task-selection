import numpy as np 
import os 
import sys 
from tqdm import tqdm

#This script takes as a parameter a directory containing the L matrices for a given dataset
#It outputs a directory replacing every matrix by its log 

allpathin = sys.argv[1]
allpathout = sys.argv[2] 


def folder_treat(pathin, pathout): 

    matricesin = os.listdir(pathin) 
    if not os.path.exists(pathout):
        os.makedirs(pathout)


    matrices = [os.path.join(pathin, x) for x in matricesin] 
    for matrix in (matricesin) :
        print(matrix)
        values = np.load(os.path.join(pathin, matrix)) 
        log = np.log(values) 
        np.save(os.path.join(pathout, matrix), log)
if not os.path.exists(allpathout):
    os.makedirs(allpathout)
for name in tqdm(os.listdir(allpathin)): 
    pathin  = os.path.join(allpathin, name)
    pathout=os.path.join(allpathout,name)
    folder_treat(pathin, pathout)

    

