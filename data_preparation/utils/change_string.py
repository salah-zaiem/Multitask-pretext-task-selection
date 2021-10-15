import pandas as pd

import os 
import sys
import tqdm 


filein = sys.argv[1]
fileout = sys.argv[2]


table_start = pd.read_csv(filein) 

wavs = table_start["wav"]
new_wavs =["csv"]*len(wavs)

table_start["csv_path_format"]=new_wavs

table_start.to_csv(fileout, index=False)
