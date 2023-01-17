
import evaluate
import librosa
import numpy as np
import torch
import torchaudio
from decord import VideoReader, cpu
from transformers import VideoMAEFeatureExtractor, Trainer, HfArgumentParser, \
    set_seed, is_torch_tpu_available, AutoConfig, AutoTokenizer
from datasets import load_dataset
from os.path import join as opj
from transformers.trainer_utils import get_last_checkpoint
import os
import logging
from utils.datacollator import DataCollator
from models.crossformer import cross_former
from arguments.data import DataTrainingArguments
from arguments.model import ModelArguments
from arguments.train import TrainArguments
import datasets
import transformers
from utils.constants import *
from utils.fea_extractor import AudioFeatureExtractor
from utils.general import sample_frame_indices, save_args

logger = logging.getLogger(__name__)

def main():


    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to warning of the Transformers logger (on main process only):
    # logger.warning(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.warning(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)
    # -----------audio-----------
    audio_config = AutoConfig.from_pretrained(model_args.audio_config_name)
    audio_config.mask_ratio = data_args.audio_mask_ratio
    audio_feature_extractor = AudioFeatureExtractor()
    audio_config.audio2spectrogram = data_args.audio2spectrogram
    audio_config.freq = data_args.freq
    audio_config.time = data_args.time
    # -----------text-----------
    text_config = AutoConfig.from_pretrained(model_args.text_config_name)
    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, cache_dir=model_args.cache_dir)
    # -----------video-----------
    video_config = AutoConfig.from_pretrained(model_args.video_config_name)
    video_feature_extractor = VideoMAEFeatureExtractor.from_pretrained(model_args.videomae_feature_extractor_name,
                                                                       size=video_config.image_size, cache_dir=model_args.cache_dir)
    # -----------fusion-----------
    fusion_config = AutoConfig.from_pretrained(model_args.fusion_config_name)

    model = cross_former(model_args, audio_config, text_config, video_config, fusion_config)

    def audio_function(examples, spectrogram=data_args.audio2spectrogram):
        def get_audio(path, target_len):
            waveform, sample_rate = librosa.load(path)
            length = len(waveform)
            if length >= target_len:
                indices = sample_frame_indices(clip_len=target_len, frame_sample_rate=1, seg_len=length)
                waveform = waveform[indices]
            else:
                waveform = np.hstack([waveform for _ in range(target_len // length)] + [waveform[:target_len % length]])
            return waveform.reshape(1, int(audio_config.image_size), int(audio_config.image_size))
        def get_spectrogram(path, target_len):
            waveform, sr = torchaudio.load(path)
            waveform = waveform - waveform.mean()
            fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr,
                                                      use_energy=False, window_type='hanning',
                                                      num_mel_bins=audio_config.freq, dither=0.0, frame_shift=10)
            n_frames = fbank.shape[0]
            # cut and pad
            if n_frames >= target_len:
                indices = sample_frame_indices(clip_len=target_len, frame_sample_rate=1, seg_len=n_frames)
                fbank = fbank[indices, :]
            else:
                m = torch.nn.ZeroPad2d((0, 0, 0, target_len-n_frames))
                fbank = m(fbank)
            return fbank.transpose(0, 1).unsqueeze(0)
        if spectrogram:
            waveforms = [get_spectrogram(opj(data_args.audio_root, i), data_args.time) for i in examples['audio']]
            return {'audio_pixel_values': waveforms}
        else:
            waveforms = [get_audio(opj(data_args.audio_root, i),  audio_config.image_size**2) for i in examples['audio']]
            pixel_values = audio_feature_extractor(waveforms=waveforms, return_tensors="pt")['pixel_values']
            return {'audio_pixel_values': pixel_values}

    if data_args.max_seq_length is None:
        max_seq_length = tokenizer.model_max_length
        if max_seq_length > 512:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 512 instead. You can change that default value by passing --max_seq_length xxx."
            )
            max_seq_length = 512
    else:
        if data_args.max_seq_length > tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
                f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            )
        max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)
    padding = "max_length" if data_args.pad_to_max_length else True
    def tokenize_function(examples):
        text = [open(opj(data_args.text_root, t), 'r').read() for t in examples['text']]
        return tokenizer(text, padding=padding, truncation=True, max_length=max_seq_length, return_attention_mask=True, return_tensors="pt")

    def video_function(examples):
        def read_video(file_path):
            videoreader = VideoReader(file_path, num_threads=1, ctx=cpu(0))
            videoreader.seek(0)
            indices = sample_frame_indices(clip_len=video_config.num_frames, frame_sample_rate=4, seg_len=len(videoreader))
            video = videoreader.get_batch(list(indices)).asnumpy()
            return video
        video = [list(read_video(opj(data_args.video_root, v))) for v in examples['video']]
        pixel_values = video_feature_extractor(video, return_tensors="pt")['pixel_values']

        return {'video_pixel_values': pixel_values}


    dataset = load_dataset('csv', data_files={'train': [data_args.train_file], 'validation': [data_args.validation_file]},
                           cache_dir=model_args.cache_dir)

    with training_args.main_process_first(desc="dataset map tokenization"):
        dataset = dataset.map(tokenize_function, batched=True)
    with training_args.main_process_first(desc="dataset map audio"):
        dataset = dataset.map(audio_function, batched=True)
    with training_args.main_process_first(desc="dataset map video"):
        dataset = dataset.map(video_function, batched=True)



    if training_args.do_train:
        if "train" not in dataset:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = dataset["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    if training_args.do_eval:
        if "validation" not in dataset:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = dataset["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

        def preprocess_logits_for_metrics(logits, labels):
            # logits ({'mask_logits': [a_mask_out['logits'], t_mask_out['logits'], v_mask_out['logits']},
            #          {'vt_cls_token': vt_cls_token}]
            t_mask_logits = logits[0][1]
            return t_mask_logits.argmax(dim=-1)

        metric = evaluate.load("accuracy")
        def ret_mask_preds_labels(preds, labels, ignore_index=-100):
            # preds have the same shape as the labels, after the argmax(-1) has been calculated
            # by preprocess_logits_for_metrics
            labels = labels.reshape(-1)
            preds = preds.reshape(-1)
            mask = labels != ignore_index
            labels = labels[mask]
            preds = preds[mask]
            return preds, labels

        def compute_metrics(eval_preds):
            # labels {id_labels, vt_match_labels, v_constrative_label, t_constrative_label}
            preds, labels = eval_preds
            mlm_preds, mlm_labels = ret_mask_preds_labels(preds, labels, -100)
            return {'VTM_acc': round(metric.compute(predictions=mlm_preds, references=mlm_labels)['accuracy'], 3)}


    data_collator = DataCollator(tokenizer=tokenizer, mlm_probability=data_args.mlm_probability,
                                 num_frames=video_config.num_frames, video_image_size=video_config.image_size, video_patch_size=video_config.patch_size,
                                 tubelet_size=video_config.tubelet_size, video_mask_ratio=data_args.video_mask_ratio)

    trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics if training_args.do_eval and not is_torch_tpu_available() else None,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics
            if training_args.do_eval and not is_torch_tpu_available() else None,
    )

    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload
        metrics = train_result.metrics
        max_train_samples = data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        # Evaluation
        if training_args.do_eval:
            metrics = trainer.evaluate()
            max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
            metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)

    save_args([model_args, data_args], output_dir=training_args.output_dir, names=[MODEL_ARGS_NAME, DATA_ARGS_NAME])

def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()

if __name__ =='__main__':
    main()
