import os 
import numpy as np 
import csv 
import sys 
import multiprocessing as mp
from tqdm import tqdm 
import pandas as pd
from functools import partial
from multiprocessing import Pool


def str_list(l): 
    return [str(x) for x in l]

def produce_line(line, path_to_csv) :
    values = line[0:-1].split(",")
    snt_id = values[0]
    #snt_id,duration,wav_path,_,_,spk_id,_,_,category_age,_,_,category_gender,_,_,accent,_,_,quality,_,_ = line.split(",")
    csv_path=(os.path.join(path_to_csv, snt_id+".csv"))
    # Composition of the csv_line
    """
    csv_line = [
        snt_id,
        str(duration),
        wav_path,
        "wav",
        "",
        spk_id,
        "string",
        "",
        str(category_age),
        "string",
        "",
        str(category_gender),
        "string",
        "",
        accent,
        "string",
        "",
        str(quality),
        "string",
        ""
    ]
    """
    csv_line = values
    csv_line.append(csv_path)
    csv_line.append("string")
    csv_line.append("")
    # Adding this line to the csv_lines list
    return csv_line
def parallel_create_proxy_csv(
    orig_tsv_file,
    path_to_extracted,
    csv_file
):
    """
    Creates the csv file given a list of wav files.

    Arguments
    ---------
    orig_tsv_file : str
        Path to the Common Voice tsv file (standard file).
    path_to_wav : str
        Path of the audio wav files.
    data_folder : str
        Path of the CommonVoice dataset.
    accented_letters : bool, optional
        Defines if accented letters will be kept as individual letters or
        transformed to the closest non-accented letters.
    duration_threshold : int
        Max duration (in seconds) to use as a threshold to filter sentences.
        The CommonVoice dataset contains very long utterance mostly containing
        noise due to open microphones.

    Returns
    -------
    None
    """

    # Check if the given files exists
    loaded_csv = open(orig_tsv_file, "r").readlines()
    first_line = loaded_csv[0][0:-1]
    loaded_csv = loaded_csv[1:]
    nb_samples = str(len(loaded_csv))


    # Adding some Prints

    csv_lines =[first_line.split(",") ]
    feature= "csv_path"
    csv_lines[0].append(feature)
    csv_lines[0].append(feature+"_format")
    csv_lines[0].append(feature+"_opts")

    # Start processing lines
    total_duration = 0.0
    print(f"len of loaded csv : {len(loaded_csv)}")

    new_csv_lines = list(tqdm(p.imap(line_function, loaded_csv),
                          total=len(loaded_csv)))
    new_csv_lines.insert(0, csv_lines[0 ])
    csv_lines = new_csv_lines
    # Writing the csv lines
    with open(csv_file, mode="w", encoding="utf-8") as csv_f:
        csv_writer = csv.writer(
            csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )

        for line in csv_lines:
            csv_writer.writerow(line)




def create_proxy_csv(
    orig_tsv_file,
    path_to_extracted,
    csv_file
):
    """
    Creates the csv file given a list of wav files.

    Arguments
    ---------
    orig_tsv_file : str
        Path to the Common Voice tsv file (standard file).
    path_to_wav : str
        Path of the audio wav files.
    data_folder : str
        Path of the CommonVoice dataset.
    accented_letters : bool, optional
        Defines if accented letters will be kept as individual letters or
        transformed to the closest non-accented letters.
    duration_threshold : int
        Max duration (in seconds) to use as a threshold to filter sentences.
        The CommonVoice dataset contains very long utterance mostly containing
        noise due to open microphones.

    Returns
    -------
    None
    """

    # Check if the given files exists
    loaded_csv = open(orig_tsv_file, "r").readlines()[1:]
    nb_samples = str(len(loaded_csv))


    # Adding some Prints

    csv_lines = [
        [
            "ID",
            "duration",
            "wav",
            "wav_format",
            "wav_opts",
            "spk_id",
            "spk_id_format",
            "spk_id_opts",
            "wrd",
            "wrd_format",
            "wrd_opts",
            "char", 
            "char_format",
            "char_opts"
        ]
    ]
    feature="csv_path"
    csv_lines[0].append(feature)
    csv_lines[0].append(feature+"_format")
    csv_lines[0].append(feature+"_opts")


    # Start processing lines
    total_duration = 0.0
    print(f"len of loaded csv : {len(loaded_csv)}")
    for line in tqdm(loaded_csv, ascii=True):
        # are located in datasets/lang/clips/
        #print(len(line.split(",")))
        #print(line.split(","))
        snt_id,duration,wav_path,_,_,spk_id,_,_,wrd,_,_,char,_,_ = line.split(",")
        read_features = pd.read_pickle(os.path.join(path_to_extracted, snt_id+".csv"))
        features_dict ={}
        # Composition of the csv_line
        csv_line = [
            snt_id,
            str(duration),
            wav_path,
            "wav",
            "",
            spk_id,
            "string",
            "",
            wrd,
            "string",
            "",
            char,
            "string",
            "",
            accent,
            "string",
            "",
            str(quality),
            "string",
            ""
        ]
        for feat in features :
            csv_line.append(features_dict[feat])
            csv_line.append("float")
            csv_line.append("")
        # Adding this line to the csv_lines list
        csv_lines.append(csv_line)

    # Writing the csv lines
    with open(csv_file, mode="w", encoding="utf-8") as csv_f:
        csv_writer = csv.writer(
            csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )

        for line in csv_lines:
            csv_writer.writerow(line)














if __name__=="__main__":
    preparation_dir = sys.argv[1]
    extraction_dir = sys.argv[2]
    outdir = sys.argv[3]
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    cpus = mp.cpu_count()
    print(f" {cpus} cpus") 
    p = Pool(cpus)
    """
    features_considered = [ 'hammarbergIndexV_sma3nz_amean',
        'F0semitoneFrom27.5Hz_sma3nz_stddevNorm',
       'F0semitoneFrom27.5Hz_sma3nz_stddevFallingSlope', 'loudness_sma3_amean',
       'loudness_sma3_stddevNorm',  'jitterLocal_sma3nz_amean',
       'jitterLocal_sma3nz_stddevNorm', 'shimmerLocaldB_sma3nz_amean',
       'shimmerLocaldB_sma3nz_stddevNorm',      'logRelF0-H1-H2_sma3nz_amean',
       'logRelF0-H1-H2_sma3nz_stddevNorm', 'logRelF0-H1-A3_sma3nz_amean',
       'logRelF0-H1-A3_sma3nz_stddevNorm', 'F1frequency_sma3nz_amean',
       'F1frequency_sma3nz_stddevNorm', 'F1bandwidth_sma3nz_amean',
       'F1bandwidth_sma3nz_stddevNorm', 'F1amplitudeLogRelF0_sma3nz_amean',
       'F1amplitudeLogRelF0_sma3nz_stddevNorm', 'F2frequency_sma3nz_amean',
       'F2frequency_sma3nz_stddevNorm', 'F2amplitudeLogRelF0_sma3nz_amean',
       'alphaRatioV_sma3nz_amean','F0semitoneFrom27.5Hz_sma3nz_amean',
       'alphaRatioV_sma3nz_stddevNorm', 'slopeV500-1500_sma3nz_stddevNorm', 'alphaRatioUV_sma3nz_amean',
       'slopeUV500-1500_sma3nz_amean', 'loudnessPeaksPerSec',
       'VoicedSegmentsPerSec', 'MeanVoicedSegmentLengthSec',
       'StddevVoicedSegmentLengthSec', 'MeanUnvoicedSegmentLength',
       'StddevUnvoicedSegmentLength']
   """
    features_considered= ["alphaRatio_sma3"]
    #features_considered=['F0final_sma', 'voicingFinalUnclipped_sma', 'jitterLocal_sma',
  #     'jitterDDP_sma', 'shimmerLocal_sma', 'logHNR_sma',
  #     'audspec_lengthL1norm_sma', 'audspecRasta_lengthL1norm_sma',
  #     'pcm_RMSenergy_sma', 'pcm_zcr_sma']
    line_function = partial(produce_line,  path_to_csv=extraction_dir)
    files = ["test.csv", "dev.csv", "train.csv"]
    for filen in files : 
        parallel_create_proxy_csv(os.path.join(preparation_dir,filen),extraction_dir,
                os.path.join(outdir, filen))
