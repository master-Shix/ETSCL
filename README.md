## Paper

The paper "ETSCL: An Evidence Theory-Based Supervised Contrastive Learning Framework for Multi-modal Glaucoma Grading" is published at the 11th OMIA Workshop on MICCAI 2024. [paper link](https://doi.org/10.1007/978-3-031-73119-8_2)

## Environment setup

1. Setup PyTorch env:
    ```bash
    conda create -n etscl python=3.8
    ```

2. Install packages:
    ```bash
    pip install -r /root/autodl-tmp/ETSCL/requirements_clean.txt
    ```

## Datasets
- GAMMA dataset: [link](https://doi.org/10.1016/j.media.2023.102938)
- Preprocessed vessel modality: [link](https://drive.google.com/file/d/1TuTXNnG-eGM8U_RQhINXHAslfrC0E6bi/view?usp=sharing)
```
datasets/
├── gamma/                          # GAMMA dataset root directory
│   ├── Glaucoma_grading/           # Glaucoma grading data directory
│   │   ├── Training/               # Training data directory
│   │   │   ├── glaucoma_grading_training_GT.xlsx  # Ground truth for glaucoma grading in the training set
│   │   │   └── multi-modality_images/  # Directory for multi-modality images used in training
│   │   │       ├── 0001/           # OCT image folder for patient 0001
│   │   │       │   ├── OCT_image_1.png  # Example of OCT image file for patient 0001
│   │   │       │   ├── OCT_image_2.png  # Example of another OCT image file for patient 0001
│   │   │       └── 0001.jpg        # Corresponding fundus image for patient 0001
│   │   │       ├── 0002/           # OCT image folder for patient 0002
│   │   │       │   ├── OCT_image_1.png  # OCT image file for patient 0002
│   │   │       └── 0002.jpg        # Corresponding fundus image for patient 0002
│   │   ├── Testing/                # Testing data directory
│   │   └── Validation/             # Validation data directory
├── Vessel/                         # Vessel dataset root directory
│   ├── Training/                   # Training data directory for vessel
│   └── Testing/                    # Testing data directory for vessel

```
## Training

### Stage 1

#### Fundus
```bash
python main_supcon_thick.py --batch_size 12 --learning_rate 0.001 --temp 0.05 --cosine --classes "fundus"
```

#### OCT
```bash
python main_supcon_thick.py --batch_size 12 --learning_rate 0.001 --temp 0.05 --cosine --classes "oct"
```
#### Vessel
```bash
python main_supcon_thick.py --batch_size 12 --learning_rate 0.001 --temp 0.05 --cosine --classes "vessel"
```
### Stage 2

```bash
 CUDA_VISIBLE_DEVICES=0,1 python main_linear2.py --batch_size 14 --learning_rate 0.002    --cosine --classes "all"  --ckpt_oct /root/autodl-tmp/SupContrast/save/SupCon/path_models/SupCon_path_resnet50_lr_0.001_decay_0.0001_bsz_14_temp_0.05_trial_0_0922_thick384_color_cosine/learning_246810/oct/ckpt_epoch_10.pth --ckpt_fundus /root/autodl-tmp/SupContrast/save/SupCon/path_models/SupCon_path_resnet50_lr_0.001_decay_0.0001_bsz_14_temp_0.05_trial_0_0922_thick384_color_cosine/learning_246810/fundus/ckpt_epoch_10.pth --ckpt_vessel /root/autodl-tmp/SupContrast/save/SupCon/path_models/SupCon_path_resnet50_lr_0.001_decay_0.0001_bsz_14_temp_0.05_trial_0_0922_thick384_color_cosine/learning_246810/vessel/ckpt_epoch_10.pth
```

## Baseline
```
baseline2.py
```

## Ablation
### main_linear_ablation3.py includes:
- CFP modality + cross-entropy classifier
- CFP+OCT modalities + cross-entropy classifier
- CFP+OCT+Vessel modalities + cross-entropy classifier


## Metrics reporting
You can copy and paste your prediction results and the ground truth in the script below, and get the kappa and accuracy reporting: 
```
trial_and_error.py
```

## Cite this paper
```
@InProceedings{10.1007/978-3-031-73119-8_2,
author="Yang, Zhiyuan
and Zhang, Bo
and Shi, Yufei
and Zhong, Ningze
and Loh, Johnathan
and Fang, Huihui
and Xu, Yanwu
and Yeo, Si Yong",
editor="Bhavna, Antony
and Chen, Hao
and Fang, Huihui
and Fu, Huazhu
and Lee, Cecilia S.",
title="ETSCL: An Evidence Theory-Based Supervised Contrastive Learning Framework for Multi-modal Glaucoma Grading",
booktitle="Ophthalmic Medical Image Analysis",
year="2025",
publisher="Springer Nature Switzerland",
address="Cham",
pages="11--21",
isbn="978-3-031-73119-8"
}
```
## Contact
If you have any question, please contact: yufei005 _AT_ e.ntu.edu.sg
