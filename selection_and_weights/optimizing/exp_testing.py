import numpy as np 
import torch
import os
import sys
from sparsemax import Sparsemax
from tqdm import tqdm

K_dir = sys.argv[1]
L_dir = sys.argv[2]

#This code outputs the conditional independance value for every considered pseudo-label
#There is no optimization involved here
# We use the speaker recognition terminology, but this is valid for every considered downstream task.
Ks = os.listdir(K_dir)
speakers = [x.split(".")[0].split("_")[-1] for x in Ks]
#Loading K matrices 
K_matrices = {}
for K in Ks : 
    sp = K.split(".")[0].split("_")[-1]
    K_matrices[sp] =torch.tensor( np.load(os.path.join(K_dir, K)))

Lmatrices ={}
features = os.listdir(L_dir)
for sp in speakers : 
    Lmatrices[sp] = []
    for feat in features : 
        pathtodirmatrix = os.path.join(L_dir, feat) 
        pathtomatrix = os.path.join(pathtodirmatrix, "L_matrix_" +sp+"_"+feat+".npy")
        Lmatrices[sp].append(torch.tensor(np.load(pathtomatrix)))
print(features)

for i in range(len(features)) :
    print(f"considered feature : {features[i]}")
    total_loss = torch.tensor([0.0])
    for speaker in tqdm(speakers) : 
        K = K_matrices[speaker] 
        Ls = Lmatrices[speaker]
        Lsum = torch.exp(Ls[i])
        sizeconsidered = K.size()[0]
        H= torch.eye(sizeconsidered) - (1/sizeconsidered**2)*torch.ones((sizeconsidered, sizeconsidered)).double()
        secondpart = torch.matmul(Lsum, H)
        firstpart = torch.matmul(K,H)
        score = (1/ (sizeconsidered**2)) * torch.trace( torch.matmul(firstpart, secondpart))
        total_loss +=score
    print(total_loss/len(speakers))
