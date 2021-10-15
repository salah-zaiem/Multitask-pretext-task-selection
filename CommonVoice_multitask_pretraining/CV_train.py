#!/usr/bin/env python3
import numpy as np
import pandas as pd
import os
import sys
from torch.nn.utils.rnn import pad_sequence
import torch

import logging
import speechbrain as sb
import torchaudio
from utils import MFCC
from hyperpyyaml import load_hyperpyyaml
from speechbrain.tokenizers.SentencePiece import SentencePiece
from speechbrain.utils.data_utils import undo_padding
from speechbrain.utils.distributed import run_on_main
from speechbrain.processing.features import STFT, Filterbank, ContextWindow,DCT, InputNormalization
from speechbrain.processing.features import spectral_magnitude, Deltas
import csv
print(sys.maxsize)
csv.field_size_limit(sys.maxsize)

"""Recipe for training a sequence-to-sequence ASR system with CommonVoice.
The system employs an encoder, a decoder, and an attention mechanism
between them. Decoding is performed with beamsearch.

To run this recipe, do the following:
> python train.py hparams/train.yaml

With the default hyperparameters, the system employs a CRDNN encoder.
The decoder is based on a standard GRU and BeamSearch (no LM).

The neural network is trained on both CTC and negative-log likelihood
targets and sub-word units estimated with Byte Pairwise Encoding (BPE).

The experiment file is flexible enough to support a large variety of
different systems. By properly changing the parameter files, you can try
different encoders, decoders, tokens (e.g, characters instead of BPE),
training languages (all CommonVoice languages), and many
other possible variations.


"""


logger = logging.getLogger(__name__)
signal_workers = ['F0semitoneFrom27.5Hz_sma3nz_amean',
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
        "alphaRatio_sma3", 
       'alphaRatioV_sma3nz_amean', "Loudness_sma3",'F0final_sma', 'voicingFinalUnclipped_sma', 'jitterLocal_sma',
       'jitterDDP_sma', 'shimmerLocal_sma', 'logHNR_sma',
       'audspec_lengthL1norm_sma', 'audspecRasta_lengthL1norm_sma',
           'pcm_RMSenergy_sma', 'pcm_zcr_sma',
       'alphaRatioV_sma3nz_stddevNorm', 'hammarbergIndexV_sma3nz_amean',
       'slopeV500-1500_sma3nz_stddevNorm', 'alphaRatioUV_sma3nz_amean',
       'slopeUV500-1500_sma3nz_amean', 'loudnessPeaksPerSec',
       'VoicedSegmentsPerSec', 'MeanVoicedSegmentLengthSec',
       'StddevVoicedSegmentLengthSec', 'MeanUnvoicedSegmentLength',
       'StddevUnvoicedSegmentLength']
   
compute_STFT = STFT(
     sample_rate=16000, win_length=25, hop_length=10, n_fft=400
    ).cuda()
 
compute_fbanks = Filterbank(n_mels=40).cuda()

compute_cw = ContextWindow(left_frames=5, right_frames=5).cuda()
compute_mfccs = DCT(input_size=40, n_out=20).cuda()

inputnorm = InputNormalization().cuda()
def MFCC(signal) : 
    features = compute_STFT(signal)
    features = spectral_magnitude(features)
    features = compute_fbanks(features)

    features = compute_mfccs(features)
#    features  = compute_cw(features)
    compute_deltas = Deltas(input_size=20).cuda()
    delta1 = compute_deltas(features)
    delta2 = compute_deltas(delta1)
    features = torch.cat([features, delta1, delta2], dim=2)
 
    return features
# Define training procedure
class ASR(sb.core.Brain):
    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""

        batch = batch.to(self.device)
        #print(batch.Loudness_sma2[0].size())
        wavs, wav_lens = batch.sig
        wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)

        self.considered_workers = self.hparams.workers
        # Forward pass
        feats = self.hparams.compute_features(wavs)
        if "gammatone" in self.considered_workers : 

            gammafeats = gammatone_filter(wavs)
            gammafeats = torch.reshape(gammafeats, (feats.size()[0], -1,64))


        else : 
            gammafeats = 0 
        feats = self.modules.normalize(feats, wav_lens)
        if "mfcc" in self.considered_workers: 
            mfcc_feats = MFCC(wavs)
            mfcc_feats = self.modules.mfcc_normalizer(mfcc_feats, wav_lens)

        else : 
            mfcc_feats = 0

        ## Add augmentation if specified
        if stage == sb.Stage.TRAIN:
            if hasattr(self.hparams, "augmentation"):
                feats = self.hparams.augmentation(feats)



        feats = self.modules.normalize(feats, wav_lens)

        dim = feats.size()[1]
        self.workers_regressors = {"age": self.modules.age, "gen" : self.modules.gen,
                                   "melfs": self.modules.dec, "accent":self.modules.accent,
                                   "quality" :self.modules.quality,
                                   'F0semitoneFrom27.5Hz_sma3nz_amean': self.modules.f0semitone,
                               'jitterLocal_sma3nz_stddevNorm' :self.modules.jitter,
                               'F1bandwidth_sma3nz_stddevNorm' :self.modules.f1std,
                               'alphaRatioV_sma3nz_amean' :self.modules.aratio,
                               'MeanVoicedSegmentLengthSec':self.modules.meanvoiced,
                                   "Loudness_sma3": self.modules.loudness,
                                   "F0final_sma" : self.modules.f0_lld,
                                   "jitterLocal_sma": self.modules.jitterLocal,
                                   "voicingFinalUnclipped_sma" :self.modules.voicing, 
                                   "alphaRatio_sma3" :self.modules.alpharatio_lld ,
                                   "pcm_zcr_sma": self.modules.pcmzcr,
                                   "audspecRasta_lengthL1norm_sma" :self.modules.audspec_rasta , 
                                   "audspec_lengthL1norm_sma0":self.modules.audspec_L1 ,
                                   "shimmerLocal_sma": self.modules.shimmerLocal,
                                   "logHNR_sma" : self.modules.loghnr,
                                   "pcm_RMSenergy_sma": self.modules.pcm,
                                   "mfcc" :  self.modules.mfcc,
                                   "gammatone" : self.modules.gammatone}
            
        # Forward pass
        # We detach as we don't need the features to be on the backward graph
        z = self.modules.enc(feats.detach())
        workers_predictions = dict()
        for worker in self.considered_workers : 
            workers_predictions[worker] = self.workers_regressors[worker](z) 
        # output layer for seq2seq log-probabilities
        if "age" in self.considered_workers : 
            workers_predictions["age"]  = torch.mean(workers_predictions["age"] , dim=1)
        if "gen" in self.considered_workers : 
            workers_predictions["gen"]  = torch.mean(workers_predictions["gen"] , dim=1)
        if "accent" in self.considered_workers : 
            workers_predictions["accent"]  = torch.mean(workers_predictions["accent"] , dim=1)
        if "quality" in self.considered_workers : 
            workers_predictions["quality"]  =torch.squeeze(torch.mean(workers_predictions["quality"] , dim=1))
        for signal_worker in signal_workers : 
            if signal_worker in self.considered_workers : 
                if signal_worker not in ["Loudness_sma3","mfcc","gammatone", "melfs", "F0final_sma",
                                         "jitterLocal_sma",  "pcm_RMSenergy_sma",
                                         "voicingFinalUnclipped_sma",
                                         "logHNR_sma", "alphaRatio_sma3",
                                        "audspecRasta_lengthL1norm_sma",
                                         "pcm_zcr_sma", "shimmerLocal_sma",
                                        "audspec_lengthL1norm_sma0"] : 
                    workers_predictions[signal_worker] =torch.mean(workers_predictions[signal_worker], dim=1)
                else :

                    workers_predictions[signal_worker]=torch.squeeze(workers_predictions[signal_worker])
        return workers_predictions, feats, mfcc_feats, gammafeats


    def compute_objectives(self, predictions, batch,other_targets, stage):
        #Load prediction 
        self.workers_losses = {"age": self.hparams.classification_loss,"gen":
                               self.hparams.classification_loss, "melfs":
                               self.hparams.reconstruction_loss, "gammatone" :
                               self.hparams.reconstruction_loss, "accent":
                               self.hparams.classification_loss,
                               "quality":self.hparams.regression_loss,
                               "mfcc" : self.hparams.reconstruction_loss}  
        for signal_worker in signal_workers : 
            if signal_worker in self.considered_workers : 
                self.workers_losses[signal_worker] =self.hparams.regression_loss
        #ids, words, word_lens = targets
        target_melf, target_mfcc, target_gammatone = other_targets
        self.workers_target={"gammatone" : target_gammatone, "melfs" : target_melf, "mfcc": target_mfcc}
        exoworkers = [x for x in self.considered_workers if x not in ["melfs","gammatone",
                                                                 "mfcc"]]
        if len(exoworkers)>0 : 

            workers_targets_values = batch.workers_targets
            Nb= len(batch.workers_targets)
            for ind,worker in enumerate(exoworkers) : 
                if worker not in ["melfs", "mfcc", "gammatone"] : 
                    worker_values = [workers_targets_values[i][worker] for i in range(Nb)]
                    worker_values = pad_sequence(worker_values)
                    self.workers_target[worker] = torch.transpose(torch.tensor(worker_values),0,1)
        workers_loss=dict()
        for worker in self.considered_workers : 
            #predictions[worker] = torch.transpose(predictions[worker],0,1)
            workers_loss[worker] = self.workers_losses[worker]()(predictions[worker],
                                                             self.workers_target[worker].to(self.device))
        self.workers_loss = workers_loss
        for worker in self.considered_workers : 
            self.workers_loss_recap[worker].append((workers_loss[worker].item()))
            if len(self.workers_loss_recap[worker])==100 : 

                outfile.write(f" for this worker : {worker} ; mean loss is {np.mean(self.workers_loss_recap[worker])}")
                outfile.write("\n")
                self.workers_loss_recap[worker] = []
        workers_weights = self.hparams.workers_weights
        for x in workers_loss : 
            if x not in workers_weights: 
                workers_weights[x] =1
        losses = [workers_loss[x] * workers_weights[x] for x in workers_loss] 
        final_loss = losses[0]
        for i in range(1, len(losses)):
            final_loss+= losses[i] 
        return final_loss, workers_loss



    def fit_batch(self, batch):
        """Train the parameters given a single batch in input"""
        self.hparams.batch_counter +=1


        predictions, feats,  mfcc, gammatone = self.compute_forward(batch, sb.Stage.TRAIN)
        other_targets = [feats, mfcc, gammatone]
        loss, workers_loss = self.compute_objectives(predictions, batch, other_targets, sb.Stage.TRAIN)
        loss.backward()
        if self.check_gradients(loss):
            self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.detach()

    def evaluate_batch(self, batch, stage):
        """Computations needed for validation/test batches"""
        predictions,feats,  mfcc, gammatone = self.compute_forward(batch, stage=stage)
        other_targets = [feats, mfcc, gammatone]
        with torch.no_grad():
            final_loss, losses = self.compute_objectives(predictions, batch,other_targets, stage=stage)
        return final_loss.detach()

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch"""
    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of an epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        self.train_stats = stage_stats

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            old_lr, new_lr = self.hparams.lr_annealing(stage_stats["loss"])
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": old_lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only()



# Define custom data procedure
def dataio_prepare(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions."""
    # 1. Define datasets
    data_folder = hparams["data_folder"]

    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_csv"], replacements={"data_root": data_folder},
    )

    if hparams["sorting"] == "ascending":
        # we sort training data to speed up training and get better results.
        train_data = train_data.filtered_sorted(
            sort_key="duration",
            key_max_value={"duration": hparams["avoid_if_longer_than"]},
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["dataloader_options"]["shuffle"] = False

    elif hparams["sorting"] == "descending":
        train_data = train_data.filtered_sorted(
            sort_key="duration",
            reverse=True,
            key_max_value={"duration": hparams["avoid_if_longer_than"]},
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["dataloader_options"]["shuffle"] = False

    elif hparams["sorting"] == "random":
        pass

    else:
        raise NotImplementedError(
            "sorting must be random, ascending or descending"
        )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["valid_csv"], replacements={"data_root": data_folder},
    )
    # We also sort the validation data so it is faster to validate
    valid_data = valid_data.filtered_sorted(sort_key="duration")

    test_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["test_csv"], replacements={"data_root": data_folder},
    )

    # We also sort the validation data so it is faster to validate
    test_data = valid_data.filtered_sorted(sort_key="duration")

    datasets = [train_data, valid_data, test_data]
    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        info = torchaudio.info(wav)
        sig = sb.dataio.dataio.read_audio(wav)
        resampled = torchaudio.transforms.Resample(
            info.sample_rate, hparams["sample_rate"],
        )(sig)
        return resampled
    needed_workers = [x for x in hparams["workers"] if x not in ["melfs",
                                                                 "gammatone", "mfcc"]]
   
    if len(needed_workers)>0 : 
        sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

        @sb.utils.data_pipeline.takes("csv_path")
        @sb.utils.data_pipeline.provides("workers_targets")
        def csv_pipeline(csv): 
            workers_values = []
            feats = {}
            csv_tab = pd.read_pickle(csv)
            for worker in needed_workers : 
                feats[worker] = torch.tensor(csv_tab[worker])
            return feats
             
 
        sb.dataio.dataset.add_dynamic_item(datasets, csv_pipeline)
        # 4. set output:
        sb.dataio.dataset.set_output_keys(
            datasets, ["id", "sig", "workers_targets"],
        )
    else : 
        sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

        sb.dataio.dataset.set_output_keys(
            datasets, ["id", "sig"],
        )
    return train_data, valid_data, test_data

if __name__ == "__main__":

    # Load hyperparameters file with command-line overrides
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # If distributed_launch=True then
    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    # Dataset preparation (parsing CommonVoice)
    from common_voice_prepare import prepare_common_voice  # noqa

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Due to DDP, we do the preparation ONLY on the main python process
    outfile = open(os.path.join(hparams["output_folder"],"losses.txt"), "a")
    outfile.write("beginning")
    outfile.write("\n")

    # Create the datasets objects as well as tokenization and encoding :-D
    train_data, valid_data, test_set = dataio_prepare(hparams)

    # Trainer initialization
    asr_brain = ASR(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts=run_opts,
        opt_class=hparams["opt_class"],
        checkpointer=hparams["checkpointer"],
    )
    asr_brain.workers_loss_recap = {}
    for worker in asr_brain.hparams.workers  : 
        asr_brain.workers_loss_recap[worker] = []
    DNNs_len = len(asr_brain.modules.enc.DNN)
    

    # removing the dropout from last layer
    asr_brain.modules.enc.DNN.block_1.dropout.p=0.0

    # Adding objects to trainer.
    #print(train_data.data)
    # Training
    asr_brain.fit(
        asr_brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=hparams["dataloader_options"],
        valid_loader_kwargs=hparams["test_dataloader_options"],
    )

    # Test

