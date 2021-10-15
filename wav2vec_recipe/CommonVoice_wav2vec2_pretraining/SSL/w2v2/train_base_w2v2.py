#!/usr/bin/env python3

#import idr_torch
import os
import hostlist
import sys
import torch
import logging
import speechbrain as sb
import torchaudio
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.data_utils import undo_padding
from speechbrain.utils.distributed import run_on_main

from transformers import Wav2Vec2Model, Wav2Vec2Config, Wav2Vec2ForPreTraining
from transformers import Wav2Vec2FeatureExtractor
from transformers.models.wav2vec2.modeling_wav2vec2 import _compute_mask_indices

# get node list from slurm
#hostnames = hostlist.expand_hostlist(os.environ["SLURM_JOB_NODELIST"])

# get IDs of reserved GPU
#gpu_ids = os.environ["SLURM_STEP_GPUS"].split(",")

# define MASTER_ADD & MASTER_PORT
#os.environ["MASTER_ADDR"] = hostnames[0]
#os.environ["MASTER_PORT"] = str(14500 + int(min(gpu_ids)))  # to avoid port conflict on the same node
#os.environ["RANK"] = str(os.environ["SLURM_PROCID"])
#os.environ["LOCAL_RANK"] = str(os.environ["SLURM_LOCALID"])

#if int(os.environ["RANK"]) == 0: 
#print(os.environ["MASTER_ADDR"])
#print(os.environ["MASTER_PORT"])
#print(os.environ["RANK"])
#et mouse print(os.environ["LOCAL_RANK"])

logger = logging.getLogger(__name__)


# Define training procedure
class ASR(sb.core.Brain):
    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""

        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)

        # if stage == sb.Stage.TRAIN:
        #    if hasattr(self.hparams, "augmentation"):
        #        wavs = self.hparams.augmentation(wavs, wav_lens)

        # compute masked indices
        batch_size, raw_sequence_length = wavs.shape
        wavs = self.modules.normalize(wavs, wav_lens)
        sequence_length = self.modules.wav2vec2.module.model._get_feat_extract_output_lengths(raw_sequence_length)
        mask_time_indices = _compute_mask_indices(
                                (batch_size, sequence_length), 
                                mask_prob=self.hparams.mask_prob, 
                                mask_length=self.hparams.mask_length,
                                device=self.device
                            )

        self.modules.wav2vec2.module.model.train()
        out = self.modules.wav2vec2(wavs, mask_time_indices, stage=sb.Stage.TRAIN)
        loss = out.loss
        if stage != sb.Stage.TRAIN:
            self.modules.wav2vec2.module.model.eval()
            with torch.no_grad():
                out = self.modules.wav2vec2(wavs, mask_time_indices, stage)

            return loss, out, mask_time_indices

        return loss

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss (CTC+NLL) given predictions and targets."""

        if stage == sb.Stage.TRAIN:
            loss = predictions
        else:
            loss, out, mask_time_indices = predictions
            cosine_sim = torch.cosine_similarity(out.projected_states, out.projected_quantized_states, dim=-1)
            acc = cosine_sim[mask_time_indices].mean()
            self.acc_metric.append(acc)

        return loss

    def fit_batch(self, batch):
        """Train the parameters given a single batch in input"""
        predictions = self.compute_forward(batch, sb.Stage.TRAIN)
        loss = self.compute_objectives(predictions, batch, sb.Stage.TRAIN)

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
        predictions = self.compute_forward(batch, stage=stage)
        with torch.no_grad():
            loss = self.compute_objectives(predictions, batch, stage=stage)
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
        else:
            stage_stats["acc"] = sum(self.acc_metric) / len(self.acc_metric)

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
                meta={"acc": stage_stats["acc"], "epoch": epoch},
                max_keys=["acc"],
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

    # 4. Set output:
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
    #run_on_main(
    #    prepare_common_voice,
    #    kwargs={
    #        "data_folder": hparams["data_folder"],
    #        "save_folder": hparams["save_folder"],
    #        "train_tsv_file": hparams["train_tsv_file"],
    #        "dev_tsv_file": hparams["dev_tsv_file"],
    #        "test_tsv_file": hparams["test_tsv_file"],
    #        "accented_letters": hparams["accented_letters"],
    #        "language": hparams["language"],
    #        "skip_prep": hparams["skip_prep"],
    #    },
    #)

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
