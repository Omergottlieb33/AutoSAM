# Learnable Prompts vs Fine-Tuning: Enhancing SAM Model Performance on Non-Natural Medical Images 

This repository contains the code for the final project in **Deep Learning in Medical Imaging**. The project, titled "Learnable Prompts vs Fine-Tuning: Enhancing SAM Model Performance on Non-Natural Medical Images," was conducted by Tal Grossman and Omer Gotlieb.

this code is is forked and highly based on [AutoSAM](https://github.com/talshaharabany/AutoSAM) repository by *Tal Shaharabany*.


## Overview

This work improves the Segment Anything Model (SAM) for medical image segmentation by replacing its conditioning mechanism with an image-based encoder. Without further fine-tuning SAM, this modification achieves state-of-the-art results on medical images and video benchmarks. In addition to AutoSam work we enhaced the work by training on non-natural images like CT and MRI

##  General knowledge
we used [PaperSpace Gradient](https://console.paperspace.com/) to train the models, and we used the following GPUs:
- A4000 (16GB) - lightwieght training and debugging
- A6000 (48GB) - for the main training
- CPU: 8 for pre-processing the data and save cost on the GPU

## Report

Please see the report attached in the submission file.

## Datasets

We used the following datasets in our experiments:

[FLAR (CT) - train](https://flare22.grand-challenge.org/Dataset/)

[LiTs (CT) - validation](https://competitions.codalab.org/competitions/17094)

[ACDC (MRI)](https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html)


## SAM checkopints

[sam base](https://drive.google.com/file/d/1ZwKc-7Q8ZaHfbGVKvvkz_LPBemxHyVpf/view?usp=drive_link)
[sam large](https://drive.google.com/file/d/16AhGjaVXrlheeXte8rvS2g2ZstWye3Xx/view?usp=drive_link)
[sam huge](https://drive.google.com/file/d/1tFYGukHxUCbCG3wPtuydO-lYakgpSYDd/view?usp=drive_link)

## Ours (Autosam) checkpoints
- [MRI (ACDC)](https://drive.google.com/file/d/1MTmZBbEQc4SuDues2Q85-tBIiQAHOES2/view?usp=sharing)
- [CT_Abd_liver (FLARE)](https://drive.google.com/file/d/1lm0QTDfqqc7vnOc9kyuyf4WX3PW5-erS/view?usp=drive_link)


## how to set the enviroment
We used 2 conda envioments for this project, one for the pre-processing the dataset and the other for the training and evaluation.
### 1. Setup enviorment for pre-processing the CT and MRI datasets
Use the conda configuration file created by the [MedSam](/home/tal/Downloads/sam.yaml) environment setup, located at `envs_configs/medsam_for_pre_processing.yml`.

Set up the environment by running the following command:

```bash
conda env create --name medsam -f envs_configs/medsam_for_pre_processing.yml
```
*Note: This enviorment will only be used when running the pre_CT_MRI dataset pre-processing script. located in `dataset/pre_CT_MR.py`*

### 2. Setup enviorment for training and evaluation env (running Autosam)
Use the conda configuration file created by SAM environment setup, located at `envs_configs/sam.yaml`.

Set up the environment by running the following command:

```bash
conda env create --name sam -f envs_configs/sam.yml
```

*Note: This enviorment will be used when running the training and evaluation scripts.*

# Usage and how to run the code

## Pre-processing the CT and MRI Datasets

1. **Activate the `medsam` environment:**
   ```bash
   conda activate medsam
   ```

2. **Run the pre-processing script:**
   For example, to pre-process the FLARE training dataset:
   ```bash
   python3 dataset/pre_CT_MR.py --task train --modality CT --anatomy Abd --nii_path /media/tal/tal_ssd/datasets/FLARE22/training/images/ --gt_path /media/tal/tal_ssd/datasets/FLARE22/training/labels/ --npy_path /media/tal/tal_ssd/datasets/npy/
   ```

### Training and Evaluation

1. **Activate the `sam` environment:**
   ```bash
   conda activate sam
   ```

2. **Run the training script:**
   For example, to train the model on the CT FLARE dataset:
   ```bash
   python3 train.py --run_name CT_Abd_liver_augmented_sam_base --task CT --epochs 10 --batch_size 10 --sam_model_type vit_b --sam_checkpoint_dir_path /mnt/share/checkpoints/sam_checkpoints/ --train_data_root /mnt/share/datasets/npy/CT_Abd_train --evaluation_data_root /mnt/share/datasets/npy/CT_Abd_val
   ```

3. **Run the evaluation script:**
   For example, to evaluate the model on the ACDC dataset:
   ```bash
   python3 evaluate.py --eval_name evaluate_acdc_sam_base_data_randomly_divided --task acdc --evaluation_data_root /mnt/share2/datasets/npy/MRI_acdc_train --model_path /home/paperspace/git/AutoSAM/results/train_acdc_sam_base_data_randomly_divided/net_last.pth --sam_model_type vit_b --sam_checkpoint_dir_path cp --num_workers 1 --output_dir results/evaluation
   ```