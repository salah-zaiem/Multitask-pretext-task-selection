import yaml
import sys
import os
filein=sys.argv[1]
results_folder=sys.argv[2]
test_name = sys.argv[3]
pase_path = sys.argv[4]
outdir = sys.argv[5]

overrides={}
with open(filein) as f : 
    lines = f.read().splitlines()
test_names =["jitterLocal_sma", "voicingFinalUnclipped_sma", "alphaRatio_sma3",
             "pcm_zcr_sma", "shimmerLocal_sma",
             "audspecRasta_lengthL1norm_sma", "pcm_RMSenergy_sma",
             "Loudness_sma3", "logHNR_sma"]

test_names = [test_name]
def get_checkpoint(name): 
    path = os.path.join(results_folder, name) 
    seed = os.listdir(path)[0]
    save_path = os.path.join(os.path.join(path,seed), "save")
    check_folder = [x for x in os.listdir(save_path) if "CKPT" in x ]
    return os.path.join(save_path, check_folder[0])
for name in test_names :
    print(get_checkpoint(name))

for name in test_names : 
    for ind, line in enumerate(lines) : 
        if line[0:20] == "embedding_model_file" : 
            start, end = line.split(":")
            lines[ind] = start + ": "+ os.path.join(get_checkpoint(name),
                                              "embedding_model.ckpt")
        if line[0:19] == "embedding_norm_file" : 
            start, end = line.split(":")
            lines[ind] = start + ": " +  os.path.join(get_checkpoint(name),
                                              "normalizer.ckpt")
        if line[0:9] == "pase_yaml" : 
            start, end = line.split(":")

            lines[ind] =start + ": "+ os.path.join("exp_params",
                                                   name+".yaml")
        if  line[0:11] == "save_folder" : 
            start, end = line.split(":") 
            lines[ind] = start + ": "+ name
        if  line[0:4] == "pase" : 
            start, end = line.split(":")
            lines[ind] = start + ": " + pase_path
    with open(os.path.join(outdir, name+".yaml"), "w") as f:
        f.write("\n".join(lines))
        f.close()

