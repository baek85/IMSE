# IMSE

Official repository of IMSE: Intrinsic Mixture of Spectral Experts Fine-tuning for Test-Time Adaptation (ICLR 2026 Poster)
Supports multiple adaptation methods including Source, TENT, CoTTA, ViDA, and IMSE.

## Requirements

- Python 3.9+
- CUDA GPU
- Conda

## Installation

```bash
conda env create -f environment.yaml
conda activate vida
```

## Data Preparation

### ImageNet-C (Required)

Download [ImageNet-C](https://github.com/hendrycks/imagenet-c) and organize it as shown below.
The `ImageNet-C` folder must be placed inside the path specified by `--data_dir`.
The default path can be changed in `conf.py` Line 269.

```
<data_dir>/
└── ImageNet-C/
    ├── gaussian_noise/
    │   ├── 1/
    │   ├── 2/
    │   ├── 3/
    │   ├── 4/
    │   └── 5/
    ├── shot_noise/
    ├── impulse_noise/
    ├── defocus_blur/
    ├── ...
    └── jpeg_compression/
```

### ImageNet (Optional, for IMSE)

IMSE requires the ImageNet validation set for source domain feature extraction.
Update the `data_dir` path in the `prepare_test_data` function at `main.py` Line 251.

### ViDA Source Model (Required for ViDA)

Download the ViDA source model from [Google Drive](https://drive.google.com/file/d/1-ft1sUROp6nb57ToLf4USifUGUguFlJF/view?usp=sharing) and place it in the `checkpoint/` folder.

## Usage

### Basic

```bash
python main.py --cfg ./cfgs/vit/source.yaml --data_dir <path_to_data> --exp_name my_exp
```

### IMSE-Retrieval

We provide two domain selection strategies for continual TTA:

**Select most similar domain, IMSE-Retrieval**

```bash
CUDA_VISIBLE_DEVICES=0 python main.py --cfg ./cfgs/vit/imse.yaml --data_dir <path_to_data> --exp_name my_exp \
    SETTING continual
```

**Domain-distance-based spectral code generation, extension of IMSE-Retrieval**

```bash
CUDA_VISIBLE_DEVICES=0 python main.py --cfg ./cfgs/vit/imse.yaml --data_dir <path_to_data> --exp_name my_exp \
    SETTING continual SVD.DYNAMIC_MODE mix_adapt SVD.TEMP 1.0
```

### Scripts

Pre-defined scripts for each setting are available in the `bash/` directory.

| Script | Setting | Methods |
|--------|---------|---------|
| `bash/0_others_ctta.sh` | C-TTA | Source, CoTTA, TENT, ViDA |
| `bash/1_imse_ctta.sh` | C-TTA | IMSE |
| `bash/2_others_recurring.sh` | Recurring | CoTTA, ViDA, TENT |
| `bash/3_imse_recurring.sh` | Recurring | IMSE |





## Project Structure

```
.
├── main.py              # Entry point
├── conf.py              # Configuration
├── environment.yaml     # Conda environment
├── bash/                # Shell scripts
├── cfgs/                # Config files
├── method/              # TTA methods (tent, cotta, vida, svd)
├── robustbench/         # Data loading, model zoo
├── timm/                # ViT models
└── ckpt/                # ViT Checkpoint
```
