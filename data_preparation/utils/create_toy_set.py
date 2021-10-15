import os
import sys 

true_set = sys.argv[1]
toy_dir = sys.argv[2]

names =["test.csv", "dev.csv"] 
if not os.path.exists(toy_dir) : 
    os.makedirs(toy_dir)
for name in names : 
    f = open(os.path.join(true_set, name), "r")
    lines = f.read().splitlines()
    interesting_lines = lines[0:20]
    g = open(os.path.join(toy_dir, name), "w")
    g.write("\n".join(interesting_lines))
    g.close()
    f.close()
