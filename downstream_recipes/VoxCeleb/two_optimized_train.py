#!/usr/bin/python3
"""Recipe for training speaker embeddings (e.g, xvectors) using the VoxCeleb Dataset.
We employ an encoder followed by a speaker classifier.

To run this recipe, use the following command:
> python train_speaker_embeddings.py {hyperparameter_file}

Using your own hyperparameter file or one of the following:
    hyperparams/train_x_vectors.yaml (for standard xvectors)
    hyperparams/train_ecapa_tdnn.yaml (for the ecapa+tdnn system)

Author
    * Mirco Ravanelli 2020
    * Hwidong Na 2020
    * Nauman Dawalatabad 2020
"""
import sys
import random
import torch
import torchaudio
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import run_on_main

from pase_train import PASE

class SpeakerBrain(sb.core.Brain):
    """Class for speaker embedding training"
    """

    def compute_forward(self, batch, stage):
        """Computation pipeline based on a encoder + speaker classifier.
        Data augmentation and environmental corruption are applied to the
        input speech.
        """
        batch = batch.to(self.device)
        wavs, lens = batch.sig

        if stage == sb.Stage.TRAIN:
            # Addding noise and reverberation
            wavs_aug = self.modules.env_corrupt(wavs, lens)

            # Adding time-domain augmentation
            wavs_aug = self.modules.augmentation(wavs_aug, lens)

            # Concatenate noisy and clean batches
            wavs = torch.cat([wavs, wavs_aug], dim=0)
            lens = torch.cat([lens, lens], dim=0)

        # Feature extraction and normalization
        feats = self.modules.compute_features(wavs)
        if self.hparams.baseline : 
            feats = self.modules.mean_var_norm(feats, lens)
        else : 
            feats = PASE_brain.modules.normalize(feats, lens)
            feats = PASE_brain.modules.enc(feats)

        # Embeddings + speaker classifier
        embeddings = self.modules.embedding_model(feats)
        outputs = self.modules.classifier(embeddings)

        return outputs, lens

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss using speaker-id as label.
        """
        predictions, lens = predictions
        uttid = batch.id
        spkid, _ = batch.spk_id_encoded

        # Concatenate labels (due to data augmentation)
        if stage == sb.Stage.TRAIN:
            spkid = torch.cat([spkid, spkid], dim=0)

        loss = self.hparams.compute_cost(predictions, spkid, lens)

        if hasattr(self.hparams.lr_annealing, "on_batch_end"):
            self.hparams.lr_annealing.on_batch_end(self.optimizer)

        if stage != sb.Stage.TRAIN:
            self.error_metrics.append(uttid, predictions, spkid, lens)

        return loss
    def fit_batch(self, batch):
        """Train the parameters given a single batch in input"""
        predictions = self.compute_forward(batch, sb.Stage.TRAIN)
        loss = self.compute_objectives(predictions, batch, sb.Stage.TRAIN)
        loss.backward()
        if self.check_gradients(loss):
            self.optimizer.step()
            PASE_optimizer.step()
        self.optimizer.zero_grad()

        PASE_optimizer.zero_grad()
        return loss.detach()

    def on_stage_start(self, stage, epoch=None):
        """Gets called at the beginning of an epoch."""
        if stage != sb.Stage.TRAIN:
            self.error_metrics = self.hparams.error_stats()

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of an epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["ErrorRate"] = self.error_metrics.summarize("average")

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            old_lr, new_lr = self.hparams.lr_annealing(epoch)
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": old_lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"ErrorRate": stage_stats["ErrorRate"]},
                min_keys=["ErrorRate"],
            )

            PASE_brain.checkpointer.save_and_keep_only()
def dataio_prep(hparams):
    "Creates the datasets and their data processing pipelines."

    data_folder = hparams["data_folder"]

    # 1. Declarations:
    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_annotation"],
        replacements={"data_root": data_folder},
    )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["valid_annotation"],
        replacements={"data_root": data_folder},
    )

    datasets = [train_data, valid_data]
    label_encoder = sb.dataio.encoder.CategoricalEncoder()

    snt_len_sample = int(hparams["sample_rate"] * hparams["sentence_len"])

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav", "start", "stop", "duration")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav, start, stop, duration):
        if hparams["random_chunk"]:
            duration_sample = int(duration * hparams["sample_rate"])
            start = random.randint(0, duration_sample - snt_len_sample - 1)
            stop = start + snt_len_sample
        else:
            start = int(start)
            stop = int(stop)
        num_frames = stop - start
        sig, fs = torchaudio.load(
            wav, num_frames=num_frames, frame_offset=start
        )
        sig = sig.transpose(0, 1).squeeze(1)
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("spk_id")
    @sb.utils.data_pipeline.provides("spk_id", "spk_id_encoded")
    def label_pipeline(spk_id):
        yield spk_id
        spk_id_encoded = label_encoder.encode_sequence_torch([spk_id])
        yield spk_id_encoded

    sb.dataio.dataset.add_dynamic_item(datasets, label_pipeline)

    # 3. Fit encoder:
    # NOTE: In this minimal example, also update from valid data
    label_encoder.update_from_didataset(train_data, output_key="spk_id")
    label_encoder.update_from_didataset(valid_data, output_key="spk_id")

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(datasets, ["id", "sig", "spk_id_encoded"])

    return train_data, valid_data, label_encoder


if __name__ == "__main__":

    # This flag enables the inbuilt cudnn auto-tuner
    torch.backends.cudnn.benchmark = True

    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[2:])
    enc_params_file,run_opts_enc,  overrides_enc = sb.parse_arguments([sys.argv[1]])
    with open(enc_params_file) as encoding_params:
        enc_params = load_hyperpyyaml(encoding_params, overrides_enc)

    PASE_brain = PASE(
            modules=enc_params["modules"],
            hparams=enc_params,
            run_opts= run_opts_enc,
            opt_class=enc_params["opt_class"],
            checkpointer=enc_params["checkpointer"],
        )

    PASE_brain.checkpointer.recover_if_possible()
    PASE_optimizer = torch.optim.SGD(PASE_brain.modules.enc.parameters(), lr=0.01, momentum=0.9)

    # Load hyperparameters file with command-line overrides
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Initialize ddp (useful only for multi-GPU DDP training)
    sb.utils.distributed.ddp_init_group(run_opts)

    # Dataset prep (parsing VoxCeleb and annotation into csv files)
    from voxceleb_prepare import prepare_voxceleb  # noqa

    run_on_main(
        prepare_voxceleb,
        kwargs={
            "data_folder": hparams["data_folder"],
            "save_folder": hparams["data_folder"],
            "splits": ["train", "dev"],
            "split_ratio": [90, 10],
            "seg_dur": int(hparams["sentence_len"]) * 100,
            "skip_prep": hparams["skip_prep"],
        },
    )

    # Dataset IO prep: creating Dataset objects and proper encodings for phones
    train_data, valid_data, label_encoder = dataio_prep(hparams)

    # Create experiment directory
    sb.core.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Brain class initialization
    speaker_brain = SpeakerBrain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # Training
    speaker_brain.fit(
        speaker_brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=hparams["dataloader_options"],
        valid_loader_kwargs=hparams["dataloader_options"],
    )
