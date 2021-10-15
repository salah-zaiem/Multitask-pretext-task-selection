import pandas as pd

import os 
import sys
import tqdm 


filein = sys.argv[1]
new_path  = sys.argv[2]
fileout = sys.argv[3]


table_start = pd.read_csv(filein) 

wavs = table_start["csv_path"]
ids = table_start["ID"]
new_wavs =[]

for ind, wav in enumerate(wavs):
    idd = ids[ind]
    new_wav = os.path.join(new_path, idd+".csv")
    new_wavs.append(new_wav)
table_start["csv_path"]=new_wavs

table_start.to_csv(fileout, index=False)
