## Overview of multimodal model
![11 9](https://user-images.githubusercontent.com/62638829/201475820-3bd24245-7aa2-48e8-8055-38a2282988f7.png)

## Prerequisites

Please refer requirements.txt.

## Running quick tests

Most examples are equipped with a mechanism to truncate the number of dataset samples to the desired length. This is useful for debugging purposes, for example to quickly check that all stages of the programs can complete, before running the same setup on the full dataset which may take hours to complete.

For example here is how to truncate to just 4 samples each:
```
python train.py \
    --max_train_samples 4 \
    --max_eval_samples 4 \
    --cache_dir cache_dir \
    --output_dir output_dir \
    --num_train_epochs 2.0 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --logging_steps 1 
    --fp16 True 
```

If all goes well, start training your data.

## Preparing you data

The file names of all modals for training can be found in `train.csv` as same as validation. Refer to `simple-multimodal/datasets` for data storage format.

Pass  `--audio_root`, `--text_root` and `--video_root` to specify the modal data path.

## Distributed training

All the PyTorch scripts mentioned above work out of the box with distributed training and mixed precision, thanks to the [Trainer API](https://huggingface.co/transformers/main_classes/trainer.html). To launch one of them on _n_ GPUs, use the following command:

```bash
python -m torch.distributed.launch \
    --nproc_per_node number_of_gpu_you_have path_to_script.py 
```

As an example, here is how you would fine-tune the model using the `train` script, with 8 GPUs:

```bash
python -m torch.distributed.launch \
    --nproc_per_node 8 train.py \
    --fusion_model_name_or_path google/vit-base-patch16-224-in21k \
    --fusion_config_name configs/crossformer/vit_config_small.json \
    --text_model_name_or_path microsoft/deberta-v2-xlarge \
    --text_config_name configs/text/debertv2_config_small.json \
    --tokenizer_name microsoft/deberta-v2-xlarge \
    --audio_model_name_or_path facebook/vit-mae-base \
    --audio_config_name configs/audio/mae_config_small.json \
    --mae_feature_extractor_name facebook/vit-mae-base \
    --video_model_name_or_path MCG-NJU/videomae-base \
    --video_config_name configs/video/videomae_config_small.json \
    --videomae_feature_extractor_name MCG-NJU/videomae-base \
    --cache_dir cache_dir \
    --audio_root datasets/audio \
    --text_root datasets/text \
    --video_root datasets/video \
    --train_file train.csv \
    --validation_file validation.csv \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --learning_rate 2e-5 \
    --num_train_epochs 50.0 \
    --warmup_ratio 0.1
    --lr_scheduler_type cosine
    --fp16 True \
    --output_dir output_dir \
    --logging_strategy steps \
    --logging_steps 500 
    [...]
```

If you have a GPU with mixed precision capabilities (architecture Pascal or more recent), you can use mixed precision training with PyTorch 1.6.0 or latest, or by installing the [Apex](https://github.com/NVIDIA/apex) library for previous versions. Just add the flag `--fp16` to your command launching one of the scripts mentioned above!

More arguments can be found in `simple-multimodal/arguments` and [Trainer API](https://huggingface.co/transformers/main_classes/trainer.html).

## Resuming training

You can resume training from a previous checkpoint like this:

1. Pass `--output_dir previous_output_dir` without `--overwrite_output_dir` to resume training from the latest checkpoint in `output_dir` (what you would use if the training was interrupted, for instance).
2. Pass `--resume_from_checkpoint path_to_a_specific_checkpoint` to resume training from that checkpoint folder.

Should you want to turn an example into a notebook where you'd no longer have access to the command line. Trainer supports resuming from a checkpoint via `trainer.train(resume_from_checkpoint)`.

1. If `resume_from_checkpoint` is `True` it will look for the last checkpoint in the value of `output_dir` passed via `TrainingArguments`.
2. If `resume_from_checkpoint` is a path to a specific checkpoint it will use that saved checkpoint folder to resume the training from.

## Changing model

1. Thanks to [Hugging Face](https://huggingface.co/docs/transformers/model_doc/bert),  you can find a config what you want to add. Making your config file and put it in `configs/text`.
2. The internal organization of each model may be different, you need to define your own encoder and decoder. please refer to `models/text/bert.py`.
3. Register the model you have added in `models/text/__init__.py`.
4. Pass `--text_model_name_or_path new_add_model_name_or_path`, `--text_config_name new_add_config_name` and `--tokenizer_name new_add_tokenizer_name`  to start training.

## Inferencing in downstream task

### Data Preparation 

1. Download Annotations and Pre-trained Checkpoints

   - [Text annotations](https://storage.googleapis.com/sfr-vision-language-research/ALPRO/data.zip)
   - [Checkpoints of pre-trained model and finetuned model](https://storage.googleapis.com/sfr-vision-language-research/ALPRO/output.zip)
   - [Externel resources](https://storage.googleapis.com/sfr-vision-language-research/ALPRO/ext.zip)
   - unzip `data.zip`, `output.zip`, `ext.zip` under `ALPRO/`.

2. Download raw videos of downstream datasets.

   - MSRVTT:

     - download train_val_videos.zip and test_videos.zip from e.g. [here](https://www.mediafire.com/folder/h14iarbs62e7p/shared).

     - check md5sum:

       ```bash
       51f2394d279cf84f1642defd9a651e6f  train_val_videos.zip
       0af68454cec9d586e92805739f3911d0  test_videos.zip
       ```

     - unzip all the videos into `data/msrvtt_ret/videos` (10k in total).

     - create the following soft link:

       ```bash
       ln -s data/msrvtt_ret/videos data/msrvtt_qa/videos```
       ```

    - MSVD:

      - download from official release:

        ```bash
        wget -nc https://www.cs.utexas.edu/users/ml/clamp/videoDescription/YouTubeClips.tar
        ```

      - check md5sum:

        ```bash
        9bdb20fcf14d59524a6febca9f6a8d89  YouTubeClips.tar
        ```

      - unzip all the videos to `data/msvd_qa/videos` (1,970 videos in total).

        ```bash
        mkdir data/msvd_qa/videos/ 
        tar xvf YouTubeClips.tar -C data/msvd_qa/videos --strip-components=1
        ```

    - DiDeMo:

      - Following [instructions](https://github.com/LisaAnne/LocalizingMoments/blob/master/README.md) and download from the official release [here](https://drive.google.com/drive/u/1/folders/1_oyJ5rQiZboipbMl6tkhY8v0s9zDkvJc);
      - unzip all the videos into `data/didemo_ret/videos`.
      - Note there might be a couple videos missing. See [here](https://github.com/LisaAnne/LocalizingMoments/blob/master/README.md#getting-the-videos) to download. However, as they account for a small portion of training set, you may feel safe to ignore.
      - Convert all the DiDeMo videos into `*.mp4` format using e.g. [`ffmpeg`](https://askubuntu.com/questions/396883/how-to-simply-convert-video-files-i-e-mkv-to-mp4).
      - We obtained 10,463 videos following these steps (with one video `77807177@N00_5753455690_1e04ccb364` missing).

  3. The directory is expected to be in the structure below:

     ```bash
     .
     |-config_release  # configuration files
     |-data  # text annotations and raw videos
     |---didemo_ret
     |-----txt
     |-----videos
     |---msrvtt_qa/...
     |---msrvtt_ret/...
     |---msvd_qa/...
     |-env  # scripts to install packages
     |-ext  # external resources, e.g. bert tokenizer
     |-output  # checkpoints for pre-trained/finetuned models
     |---downstreams
     |-----didemo_ret
     |-------public
     |---------ckpt # official finetuned checkpoints
     |---------log # inference log
     |---------results_test
     |-----------step_best_1_mean
     |-----msrvtt_qa/...
     |-----msrvtt_ret/...
     |-----msvd_qa/...
     |-run_scripts  # bash scripts to launch experiments
     |-src  # source code
     ```

### Inference with Official Checkpoints

  ```bash
  cd run_scripts
  bash inf_msrvtt_ret.sh
  # {'text2video': {'r1': 33.9, 'r5': 60.7, 'r10': 73.2, 'medianR': 3.0, 'meanR': 27.404}}
  bash inf_didemo_ret.sh
  # {'text2video': {'r1': 35.9, 'r5': 67.5, 'r10': 78.8, 'medianR': 3.0, 'meanR': 19.125}}
  bash inf_msrvtt_qa.sh
  # {'ratios': {'what_ratio': [68.48, 49872], 'who_ratio': [27.99, 20385], 'how_ratio': [2.25, 1640], 'where_ratio': [0.34, 250], 'when_ratio': [0.93, 677]}, 'overall_acc': 42.12, 'what_acc': 36.05, 'who_acc': 52.24, 'how_acc': 85.67, 'where_acc': 42.8, 'when_acc': 78.88}
  bash inf_msvd_qa.sh
  # {'ratios': {'what_ratio': [61.93, 8150], 'who_ratio': [34.6, 4554], 'how_ratio': [2.81, 370], 'where_ratio': [0.21, 28], 'when_ratio': [0.44, 58]}, 'overall_acc': 45.91, 'what_acc': 37.02, 'who_acc': 58.59, 'how_acc': 81.62, 'where_acc': 46.43, 'when_acc': 72.41}
  ```


## 