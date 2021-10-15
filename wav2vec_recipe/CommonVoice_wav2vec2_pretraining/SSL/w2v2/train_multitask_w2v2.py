#!/usr/bin/env python3
import sys
import numpy as np
import pandas as pd
import torch
import logging
import speechbrain as sb
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.data_utils import undo_padding
from speechbrain.utils.distributed import run_on_main
from speechbrain.processing.features import STFT, Filterbank, ContextWindow,DCT, InputNormalization
from speechbrain.processing.features import spectral_magnitude, Deltas

from transformers import Wav2Vec2Model, Wav2Vec2Config, Wav2Vec2ForPreTraining
from transformers import Wav2Vec2FeatureExtractor
from transformers.models.wav2vec2.modeling_wav2vec2 import _compute_mask_indices

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
    compute_deltas = Deltas(input_size=20).cuda()
    delta1 = compute_deltas(features)
    delta2 = compute_deltas(delta1)
    features = torch.cat([features, delta1, delta2], dim=2)
 
    return features


# Define training procedure
class ASR(sb.core.Brain):
    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""
        self.considered_workers = self.hparams.workers
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)

        feats = 0
    
        # feats = self.modules.normalize(feats, wav_lens)
        # compute masked indices
        batch_size, raw_sequence_length = wavs.shape
        wavs = self.modules.normalize_wavs(wavs, wav_lens)
        sequence_length = self.modules.wav2vec2.module.model._get_feat_extract_output_lengths(raw_sequence_length)
        mask_time_indices = _compute_mask_indices((batch_size, sequence_length), mask_prob=0.65, mask_length=10,
                                                  device=self.device)
        if "mfcc" in self.considered_workers: 
            mfcc_feats = MFCC(wavs)
            mfcc_feats = self.modules.mfcc_normalizer(mfcc_feats, wav_lens)

        else : 
            mfcc_feats = 0

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
                                   "mfcc" :  self.modules.mfcc}
            
        # Forward pass
        # We detach as we don't need the features to be on the backward graph



        self.modules.wav2vec2.module.model.train()
        out = self.modules.wav2vec2(wavs, mask_time_indices, stage=sb.Stage.TRAIN)
        loss = out.loss 
        embeddings = out.hidden_states[0]
        if stage != sb.Stage.TRAIN:
            self.modules.wav2vec2.module.model.eval()
            with torch.no_grad():
                out = self.modules.wav2vec2(wavs, mask_time_indices, stage)

            return loss, out, mask_time_indices
        z = embeddings
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
        for signal_worker in signal_workers + ["melfs", "mfcc"] : 
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
                    act_tensor = workers_predictions[signal_worker]
                    act_shape = act_tensor.shape
                    act_tensor =torch.reshape(act_tensor, (act_shape[0], act_shape[1]*2, act_shape[2]//2)) 
                    workers_predictions[signal_worker]=torch.squeeze(act_tensor)
        return loss, workers_predictions, batch, feats, mfcc_feats, stage

    def compute_objectives(self,wav2vec_loss, predictions, batch,mel, mfcc, stage):
        """Computes the loss (CTC+NLL) given predictions and targets."""
        self.workers_losses = {"age": self.hparams.classification_loss,"gen":
                               self.hparams.classification_loss, "melfs":
                               self.hparams.reconstruction_loss, "accent":
                               self.hparams.classification_loss,
                               "quality":self.hparams.regression_loss,
                               "mfcc" : self.hparams.reconstruction_loss}  
        for signal_worker in signal_workers : 
            if signal_worker in self.considered_workers : 
                self.workers_losses[signal_worker] =self.hparams.regression_loss
        #ids, words, word_lens = targets
        target_melf, target_mfcc = mel, mfcc
        self.workers_target={ "melfs" : target_melf, "mfcc": target_mfcc}
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

            if self.workers_target[worker].shape[1]  != predictions[worker].shape[1] : 
                if len(self.workers_target[worker].shape) == 3 : 
                    number = self.workers_target[worker].shape[1] 

                    diff = number - predictions[worker].shape[1] 
                    if diff > 0 :

                        act_tensor = predictions[worker]
                        new_tensor = torch.cat((act_tensor, torch.zeros(act_tensor.shape[0],diff,act_tensor.shape[2]).to(self.device)), 1) 
                        predictions[worker] = new_tensor
                    if diff < 0 : 
                        act_tensor = self.workers_target[worker]
                        new_tensor = torch.cat((act_tensor, torch.zeros(act_tensor.shape[0],-1*diff,act_tensor.shape[2]).to(self.device)), 1) 
                        self.workers_target[worker] = new_tensor

                else : 
                    number = self.workers_target[worker].shape[1] 
                    diff = number - predictions[worker].shape[1] 
                    if diff > 0 :

                        act_tensor = predictions[worker]
                        new_tensor = torch.cat((act_tensor.to(self.device), torch.zeros(act_tensor.shape[0],diff).to(self.device)), 1) 
                        predictions[worker] = new_tensor
                    if diff < 0 : 

                        act_tensor = self.workers_target[worker]
                        new_tensor = torch.cat((act_tensor.to(self.device), torch.zeros(act_tensor.shape[0],-1*diff).to(self.device)), 1) 
                        self.workers_target[worker] = new_tensor


            workers_loss[worker] = self.workers_losses[worker]()(predictions[worker],
                                                             self.workers_target[worker].to(self.device))
            workers_loss[worker] = workers_loss[worker]* self.workers_target[worker].shape[1]
        self.workers_loss = workers_loss
        for worker in self.considered_workers : 
            self.workers_loss_recap[worker].append((workers_loss[worker].item()))
            if len(self.workers_loss_recap[worker])==1000 : 
                if sb.utils.distributed.if_main_process():
                    logger.info(f" for this worker : {worker} ; mean loss is {np.mean(self.workers_loss_recap[worker])}")
                    self.workers_loss_recap[worker] = []
        losses = [workers_loss[x] for x in workers_loss] 
        final_loss = losses[0]
        for i in range(1, len(losses)):
            final_loss+= losses[i] 
        
        if stage == sb.Stage.TRAIN:
            #print(f" wav2vec loss =  {wav2vec_loss}")
            loss = final_loss + wav2vec_loss
        else:
            loss, out, mask_time_indices = wav2vec_loss
            cosine_sim = torch.cosine_similarity(out.projected_states, out.projected_quantized_states, dim=-1)
            acc = cosine_sim[mask_time_indices].mean()
            print(f"self.acc")
            self.acc_metric.append(acc)

        return loss

    def fit_batch(self, batch):
        """Train the parameters given a single batch in input"""
        wav2vec_loss, predictions, batch, mel, mfcc, stage = self.compute_forward(batch, sb.Stage.TRAIN)
        loss = self.compute_objectives(wav2vec_loss,predictions,batch, mel, mfcc,  sb.Stage.TRAIN)

        # normalize the loss by gradient_accumulation step
        (loss / self.hparams.gradient_accumulation).backward()

        if self.step % self.hparams.gradient_accumulation == 0:
            # gradient clipping & early stop if loss is not fini
            self.check_gradients(loss)

            self.optimizer.step()
            self.optimizer.zero_grad()

            # anneal lr every update
            self.hparams.noam_annealing(self.optimizer)

        return loss.detach()

    def evaluate_batch(self, batch, stage):
        """Computations needed for validation/test batches"""

        wav2vec_loss, predictions,batch, mel, mfcc, stage = self.compute_forward(batch, sb.Stage.TRAIN)
        with torch.no_grad():
            loss = self.compute_objectives(wav2vec_loss, predictions, batch,mel, mfcc, stage=stage)
        return loss.detach()

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch"""
        if stage != sb.Stage.TRAIN:
            self.acc_metric = []

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of an epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        #else:
        #    stage_stats["acc"] = sum(self.acc_metric) / len(self.acc_metric)
        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID and sb.utils.distributed.if_main_process():
            lr = self.hparams.noam_annealing.current_lr
            steps = self.hparams.noam_annealing.n_steps
            optimizer = self.optimizer.__class__.__name__

            epoch_stats = {
                "epoch": epoch,
                "lr": lr,
                "steps": steps,
                "optimizer": optimizer,
            }
            self.hparams.train_logger.log_stats(
                stats_meta=epoch_stats,
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"loss": stage_stats["loss"], "epoch": epoch},
                min_keys=["loss"],
            )

        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            # with open(self.hparams.wer_file, "w") as w:
            #     self.wer_metric.write_stats(w)

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
            key_min_value={"duration": hparams["avoid_if_shorter_than"]},
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["dataloader_options"]["shuffle"] = False

    elif hparams["sorting"] == "descending":
        train_data = train_data.filtered_sorted(
            sort_key="duration",
            reverse=True,
            key_max_value={"duration": hparams["avoid_if_longer_than"]},
            key_min_value={"duration": hparams["avoid_if_shorter_than"]},
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["dataloader_options"]["shuffle"] = False

    elif hparams["sorting"] == "random":
        train_data = train_data.filtered_sorted(
            key_max_value={"duration": hparams["avoid_if_longer_than"]},
            key_min_value={"duration": hparams["avoid_if_shorter_than"]},
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["dataloader_options"]["shuffle"] = False

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
    test_data = test_data.filtered_sorted(sort_key="duration")

    datasets = [train_data, valid_data, test_data]

    # defining tokenizer and loading it

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        info = torchaudio.info(wav)
        sig = sb.dataio.dataio.read_audio(wav)
        if info.num_channels > 1:
            sig = torch.mean(sig, dim=1)
        resampled = torchaudio.transforms.Resample(
            info.sample_rate, hparams["sample_rate"],
        )(sig)
        return resampled

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

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
   # run_on_main(
   #     prepare_common_voice,
   #     kwargs={
   #         "data_folder": hparams["data_folder"],
   #         "save_folder": hparams["save_folder"],
   #         "train_tsv_file": hparams["train_tsv_file"],
   #         "dev_tsv_file": hparams["dev_tsv_file"],
   #         "test_tsv_file": hparams["test_tsv_file"],
   #         "accented_letters": hparams["accented_letters"],
   #         "language": hparams["language"],
   #         "skip_prep": hparams["skip_prep"],
   #     },
   # )

    # Create the datasets objects as well as tokenization and encoding :-D
    train_data, valid_data, test_data = dataio_prepare(hparams)

    # Trainer initialization
    asr_brain = ASR(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts=run_opts,
        opt_class=hparams["opt_class"],
        checkpointer=hparams["checkpointer"],
    )
    # Adding objects to trainer.
    asr_brain.workers_loss_recap = {}
    for worker in asr_brain.hparams.workers  : 
        asr_brain.workers_loss_recap[worker] = []

    # Training
    asr_brain.fit(
        asr_brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=hparams["dataloader_options"],
        valid_loader_kwargs=hparams["test_dataloader_options"],
    )

    # Test
    asr_brain.hparams.wer_file = hparams["output_folder"] + "/wer_test.txt"
    asr_brain.evaluate(
        test_data,
        min_key="WER",
        test_loader_kwargs=hparams["test_dataloader_options"],
    )
