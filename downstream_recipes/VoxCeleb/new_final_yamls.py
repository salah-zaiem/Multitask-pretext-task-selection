import yaml
import sys
import os
filein=sys.argv[1]
embedding_path = sys.argv[2]
outdir = sys.argv[3]
overrides={}
with open(filein) as f : 
    lines = f.read().splitlines()
test_names =["jitterLocal_sma", "voicingFinalUnclipped_sma", "alphaRatio_sma3",
             "pcm_zcr_sma", "shimmerLocal_sma",
             "audspecRasta_lengthL1norm_sma", "pcm_RMSenergy_sma",
             "Loudness_sma3", "logHNR_sma"]
test_names = [embedding_path]
for name in test_names : 
    for ind, line in enumerate(lines) : 
        if line[0:15] == "embedding_param" : 
            print(line)
            start, end = line.split(":")

            lines[ind] =start + ": "+ os.path.join(name)
        if  line[0:13] == "output_folder" : 
            start, end = line.split(":") 
            lines[ind] = start + ": results/voxceleb1/verif_"+ name
    with open(os.path.join(outdir, name.split("/")[-1]), "w") as f:
        f.write("\n".join(lines))
        f.close()

