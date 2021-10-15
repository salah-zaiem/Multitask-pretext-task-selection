"""
Data preparation.

Download: https://voice.mozilla.org/en/datasets

Author
------
Titouan Parcollet
"""
from tqdm import tqdm
import os
import sys
import csv
import re
import logging
import torchaudio
import unicodedata
from tqdm.contrib import tzip

logger = logging.getLogger(__name__)


def prepare_VoxPopuli(
    data_folder,
    save_folder,
    duration_threshold=31,
    language="en",
    skip_prep=False,
):
    """
    Prepares the csv files for the Mozilla Common Voice dataset.
    Download: https://voice.mozilla.org/en/datasets

    Arguments
    ---------
    data_folder : str
        Path to the folder where the original cut VoxPopuli dataset is stored.
        This path should include the lang:/datasets/voxpopuli/real_unlabeled/en
    save_folder : str
        The directory where to store the csv files.
    train_tsv_file : str, optional
        Path to the Train Common Voice .tsv file (cs)
    dev_tsv_file : str, optional
        Path to the Dev Common Voice .tsv file (cs)
    test_tsv_file : str, optional
        Path to the Test Common Voice .tsv file (cs)
    accented_letters : bool, optional
        Defines if accented letters will be kept as individual letters or
        transformed to the closest non-accented letters.
    duration_threshold : int, optional
        Max duration (in seconds) to use as a threshold to filter sentences.
        The CommonVoice dataset contains very long utterance mostly containing
        noise due to open microphones.
    skip_prep: bool
        If True, skip data preparation.

    Example
    -------
    >>> from recipes.CommonVoice.common_voice_prepare import prepare_common_voice
    >>> data_folder = '/datasets/CommonVoice/en'
    >>> save_folder = 'exp/CommonVoice_exp'
    >>> train_tsv_file = '/datasets/CommonVoice/en/dev.tsv'
    >>> dev_tsv_file = '/datasets/CommonVoice/en/dev.tsv'
    >>> test_tsv_file = '/datasets/CommonVoice/en/test.tsv'
    >>> accented_letters = False
    >>> duration_threshold = 10
    >>> prepare_common_voice( \
                 data_folder, \
                 save_folder, \
                 train_tsv_file, \
                 dev_tsv_file, \
                 test_tsv_file, \
                 accented_letters, \
                 duration_threshold, \
                 language="en" \
                 )
    """

    if skip_prep:
        return

    # If not specified point toward standard location w.r.t CommonVoice tree
    # Setting the save folder
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Setting ouput files
    save_csv_train = save_folder + "/train.csv"
    save_csv_dev = save_folder + "/dev.csv"
    save_csv_test = save_folder + "/test.csv"

    # If csv already exists, we skip the data preparation
    if skip(save_csv_train, save_csv_dev, save_csv_test):

        msg = "%s already exists, skipping data preparation!" % (save_csv_train)
        logger.info(msg)

        msg = "%s already exists, skipping data preparation!" % (save_csv_dev)
        logger.info(msg)

        msg = "%s already exists, skipping data preparation!" % (save_csv_test)
        logger.info(msg)

        return

    # Additional checks to make sure the data folder contains Common Voice
    check_voxpopuli_folders(data_folder)

    # Creating csv file for training data

    create_csv(
            data_folder,
            save_csv_train,
    )
'''
    # Creating csv file for dev data
    if dev_tsv_file is not None:

        create_csv(
            dev_tsv_file, save_csv_dev, data_folder, accented_letters, language,
        )

    # Creating csv file for test data
    if test_tsv_file is not None:

        create_csv(
            test_tsv_file,
            save_csv_test,
            data_folder,
            accented_letters,
            language,
        )
'''

def skip(save_csv_train, save_csv_dev, save_csv_test):
    """
    Detects if the Common Voice data preparation has been already done.

    If the preparation has been done, we can skip it.

    Returns
    -------
    bool
        if True, the preparation phase can be skipped.
        if False, it must be done.
    """

    # Checking folders and save options
    skip = False

    if (
        os.path.isfile(save_csv_train)
        and os.path.isfile(save_csv_dev)
        and os.path.isfile(save_csv_test)
    ):
        skip = True

    return skip


def create_csv(
    data_folder, csv_file):
    """
    Creates the csv file given a list of wav files.

    Arguments
    ---------
    orig_tsv_file : str
        Path tof the Common Voice tsv file (standard file).
    data_folder : str
        Path of the CommonVoice dataset.
    accented_letters : bool, optional
        Defines if accented letters will be kept as individual letters or
        transformed to the closest non-accented letters.

    Returns
    -------
    None
    """

    folders = os.listdir(data_folder) 
    folders_path = [os.path.join(data_folder, x) for x in folders]
    nb_samples = sum([len(os.listdir(x)) for x in folders_path])

    msg = "Preparing CSV files for %s samples ..." % (str(nb_samples))
    logger.info(msg)

    # Adding some Prints
    msg = "Creating csv lists in %s ..." % (csv_file)
    logger.info(msg)

    csv_lines = [
        [
            "ID",
            "duration",
            "wav",
            "wav_format",
            "wav_opts",
            "year",
            "year_format",
            "year_opts",
        ]
    ]

    # Start processing lines
    total_duration = 0.0
    for ind, folder in enumerate(folders_path) : 
        files_folder = os.listdir(folder)
        for filein in tqdm(files_folder) : 

            year =folders[ind] 
        # Path is at indice 1 in Common Voice tsv files. And .mp3 files
        # are located in datasets/lang/clips/
            ogg_path = os.path.join(folder, filein)
            file_name = ogg_path.split("/")[-1].split(".")[0]
            snt_id = file_name

            # Reading the signal (to retrieve duration in seconds)
            if os.path.isfile(ogg_path):
                info = torchaudio.info(ogg_path)
            else:
                msg = "\tError loading: %s" % (str(len(file_name)))
                logger.info(msg)
                continue
            duration = info[0].length / info[0].rate
            total_duration += duration

            # Composition of the csv_line
            csv_line = [
                snt_id,
                str(duration),
                ogg_path,
                "wav",
                "",
                year, 
                "string",
                "",
            ]

        # Adding this line to the csv_lines list
            csv_lines.append(csv_line)

    # Writing the csv lines
    with open(csv_file, mode="w", encoding="utf-8") as csv_f:
        csv_writer = csv.writer(
            csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )

        for line in csv_lines:
            csv_writer.writerow(line)

    # Final prints
    msg = "%s sucessfully created!" % (csv_file)
    logger.info(msg)
    msg = "Number of samples: %s " % (str(nb_samples))
    logger.info(msg)
    msg = "Total duration: %s Hours" % (str(round(total_duration / 3600, 2)))
    logger.info(msg)


def check_voxpopuli_folders(data_folder):
    """
    Check if the data folder actually contains the Common Voice dataset.

    If not, raises an error.

    Returns
    -------
    None

    Raises
    ------
    FileNotFoundError
        If data folder doesn't contain Common Voice dataset.
    """

    files_str = "/2009/"

    # Checking clips
    if not os.path.exists(data_folder + files_str):

        err_msg = (
            "the folder %s does not exist (it is expected in "
            "the Common Voice dataset)" % (data_folder + files_str)
        )
        raise FileNotFoundError(err_msg)
if __name__=="__main__": 
    data_folder = sys.argv[1]
    csv_folder = sys.argv[2]
    prepare_VoxPopuli(data_folder, csv_folder)
