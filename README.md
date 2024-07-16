# README

## 环境安装

1. 创建一个PyTorch环境：
    ```bash
    conda create -n etscl python=3.8
    ```

2. 安装所需要的包：
    ```bash
    pip install -r /root/autodl-tmp/ETSCL/requirements_clean.txt
    ```

## 数据集准备

请下载数据集并放置在指定目录下：[下载链接](xxx)

## 运行代码

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