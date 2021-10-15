import numpy as np 
import torch
import os
import sys
from sparsemax import Sparsemax
from tqdm import tqdm
import time
from datetime import datetime
K_dir = sys.argv[1] #K matrices dir

L_dir = sys.argv[2] #L matrices dir
verbose = int(sys.argv[3]) 
outdir = sys.argv[4] # Dir where results will be output

# datetime object containing current date and time
now = datetime.now()

dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
dt_string=dt_string.replace(" ", "")
dt_string=dt_string.replace("/", "-")
dt_string=dt_string.replace(":", "-")
n_epochs = 1000
outfile = os.path.join(outdir, dt_string)
f = open(outfile, "w")
Ks = os.listdir(K_dir)

downstream_classes = [x.split(".")[0].split("_")[-1] for x in Ks]

#Loading K matrices 
K_matrices = {}
for K in Ks : 
    sp = K.split(".")[0].split("_")[-1]
    K_matrices[sp] =torch.tensor( np.load(os.path.join(K_dir, K)))
#Loading L matrices
Lmatrices ={}
features = os.listdir(L_dir)
for sp in downstream_classes : 
    Lmatrices[sp] = []
    for feat in features : 
        pathtodirmatrix = os.path.join(L_dir, feat) 
        pathtomatrix = os.path.join(pathtodirmatrix, "L_matrix_" +sp+"_"+feat+".npy")
        Lmatrices[sp].append(torch.tensor(np.load(pathtomatrix)))
print(features)
epsilon  = 0.05 # Noise in sparsemax initialization

#Function choice, put sparse = True, if you want the sparsemax option.
feats_number = len(features)
function=torch.nn.Softmax
sparse = True 
if sparse : 
    function = Sparsemax
W = torch.nn.Parameter(torch.randn(1,len(features)))
if sparse : 
    #For sparsemax, we initialize uniformly with a little noise (function of epsilon defined above )
    v = torch.ones(1,feats_number) + torch.randn(1,feats_number) *epsilon
    W = torch.nn.Parameter(v)
W.requires_grad = True

optimizer = torch.optim.SGD([W], lr=0.01, momentum=0.9)
function = function(dim=-1)

sigma = 0 # Norm 2 penality

for i in tqdm(range(n_epochs)) :
    optimizer.zero_grad()
    total_loss = torch.tensor([0.0])
    lambdas = function(W)
    
    if verbose : 
        print(lambdas)
        print(features)
    for speaker in (downstream_classes) : 
        K = K_matrices[speaker] 
        Ls = Lmatrices[speaker]
        Lsum = torch.zeros(K.size()[0]) 
        for ind in range(len(Ls)) : 
            Ls[ind] = torch.clamp(Ls[ind], -10,0)
            Lsum  = Lsum +  lambdas[0,ind] *Ls[ind]

        Lsum = torch.exp(Lsum)
        sizeconsidered = K.size()[0]
        H= torch.eye(sizeconsidered) - (1/sizeconsidered**2)*torch.ones((sizeconsidered, sizeconsidered)).double()
        secondpart = torch.matmul(Lsum, H)
        firstpart = torch.matmul(K,H)
        score = (1/ (sizeconsidered**2)) * torch.trace( torch.matmul(firstpart, secondpart))
        total_loss +=score
    if verbose  :
        print(f"HCIS loss = {total_loss/len(downstream_classes)}")
    total_loss = total_loss / len(downstream_classes) 
    #If sigma !=0 we penalize the norm, not used in the paper, but maybe useful to avoid one pseudo-label selection. 
    total_loss += torch.norm(lambdas) * sigma
    if verbose : 
        print(f" norm of lambdas: {torch.norm(lambdas)}")
    #Update the W
    total_loss.backward()
    optimizer.step()
    
f.write(str(features))
f.write("\n")
f.write(str(lambdas))
f.write(str(total_loss))
print(lambdas)
print(f" end of training loss : {total_loss}")
f.close()
