import os 
import numpy as np 
import csv 
import sys 
import multiprocessing as mp
from tqdm import tqdm 
import pandas as pd
from functools import partial
from multiprocessing import Pool
import pandas as pd

values = ['F0final_sma', "audspecRasta_lengthL1norm_sma", "pcm_RMSenergy_sma",
         "pcm_zcr_sma", "voicingFinalUnclipped_sma", "logHNR_sma"]
values.append("alphaRatio_sma3")


def file_treatment_lld(filepath, outdir, feats_means, feats_std) : 
    cv_points = pd.read_pickle(filepath)
    fname = filepath.split("/")[-1]
    for value in (values) : 
        cv_points[value] = (cv_points[value]  - feats_means[value] ) / feats_std[value]
    cv_points.to_pickle(os.path.join(outdir, fname))

def get_mean_and_std(files) :
    feats_means = {}
    feats_std= {}
    feats_values= {}
    for val in values : 
        feats_values[val] = []
    for f in tqdm(files) : 
        read_file= pd.read_pickle(f)
        columns =read_file.columns
        for val in values : 
            feats_values[val] += list(read_file[val])
    for val in values : 
        feats_means[val] = np.mean(feats_values[val])
        feats_std[val] = np.std(feats_values[val])
    return feats_means, feats_std






if __name__=="__main__":
    preparation_dir = sys.argv[1]
    outdir = sys.argv[2]
    if not os.path.exists(outdir): 
        os.makedirs(outdir) 
    files = os.listdir(preparation_dir) 
    files = [os.path.join(preparation_dir, x) for x in files]
    means_files = files[0:20000]
    means, stds = get_mean_and_std(means_files)
    part_func = partial(file_treatment_lld,outdir = outdir, feats_means = means, feats_std= stds) 
    p = Pool(mp.cpu_count())

    r = list(tqdm(p.imap(part_func, files), total=len(files)))
