import os
import sys
from tqdm import tqdm



filein = sys.argv[1]
dircheck = sys.argv[2]
fileout = sys.argv[3]


p = open(filein, "r")
t = p.read().splitlines()
first_line = t[0]
new_lines=[]
set_elements = set(os.listdir(dircheck))
for line in tqdm(t[1:]) : 
    element = line.split(",")[0]
    if (element+".csv") in set_elements :
        new_lines.append(line)

out = open(fileout, "w")
out.write(first_line)
out.write("\n") 
for line in tqdm(new_lines) : 
    out.write(line)
    out.write("\n")

out.close()
p.close()



