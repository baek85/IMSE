# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Configuration file (powered by YACS)."""

import argparse
import os
import sys
import logging
import random
import torch
import numpy as np
from datetime import datetime
from iopath.common.file_io import g_pathmgr
from yacs.config import CfgNode as CfgNode


# Global config object (example usage: from core.config import cfg)
_C = CfgNode()
cfg = _C


# ----------------------------- Model options ------------------------------- #
_C.MODEL = CfgNode()

# Check https://github.com/RobustBench/robustbench for available models
# _C.MODEL.ARCH = 'Standard'
_C.MODEL.ARCH = 'VITB_augreg_in21k_ours'

# Choice of (source, norm, tent)
# - source: baseline without adaptation
# - norm: test-time normalization
# - tent: test-time entropy minimization (ours)
_C.MODEL.ADAPTATION = 'source'

# By default tent is online, with updates persisting across batches.
# To make adaptation episodic, and reset the model for each batch, choose True.
_C.MODEL.EPISODIC = False

# ----------------------------- Corruption options -------------------------- #
_C.CORRUPTION = CfgNode()

# Dataset for evaluation
_C.CORRUPTION.DATASET = 'cifar10'

# Check https://github.com/hendrycks/robustness for corruption details
_C.CORRUPTION.TYPE = ['gaussian_noise', 'shot_noise', 'impulse_noise',
                      'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
                      'snow', 'frost', 'fog', 'brightness', 'contrast',
                      'elastic_transform', 'pixelate', 'jpeg_compression']
_C.CORRUPTION.SEVERITY = [5, 4, 3, 2, 1]

# Number of examples to evaluate 
# The 5000 val images defined by Robustbench were actually used:
# Please see https://github.com/RobustBench/robustbench/blob/7af0e34c6b383cd73ea7a1bbced358d7ce6ad22f/robustbench/data/imagenet_test_image_ids.txt
_C.CORRUPTION.NUM_EX = 5000

# ------------------------------- Batch norm options ------------------------ #
_C.BN = CfgNode()

# BN epsilon
_C.BN.EPS = 1e-5

# BN momentum (BN momentum in PyTorch = 1 - BN momentum in Caffe2)
_C.BN.MOM = 0.1

# ------------------------------- Optimizer options ------------------------- #
_C.OPTIM = CfgNode()

# Number of updates per batch
_C.OPTIM.STEPS = 1

# Learning rate
_C.OPTIM.LR = 1e-3

# Choices: Adam, SGD
_C.OPTIM.METHOD = 'Adam'

# Beta
_C.OPTIM.BETA = 0.9

# Momentum
_C.OPTIM.MOMENTUM = 0.9

# Momentum dampening
_C.OPTIM.DAMPENING = 0.0

# Nesterov momentum
_C.OPTIM.NESTEROV = True

# L2 regularization
_C.OPTIM.WD = 0.0

# ------------------------------- Testing options --------------------------- #
_C.TEST = CfgNode()

# Batch size for evaluation (and updates for norm + tent)
_C.TEST.BATCH_SIZE = 128

# --------------------------------- CUDNN options --------------------------- #
_C.CUDNN = CfgNode()

# Benchmark to select fastest CUDNN algorithms (best for fixed input sizes)
_C.CUDNN.BENCHMARK = True

# ---------------------------------- Misc options --------------------------- #

# Optional description of a config
_C.DESC = ""

# Note that non-determinism is still present due to non-deterministic GPU ops
_C.RNG_SEED = 1

# Output directory
_C.SAVE_DIR = "./output"

# Data directory
_C.DATA_DIR = "/root/data01"

# Weight directory
_C.CKPT_DIR = "./ckpt"

# Log destination (in SAVE_DIR)
_C.LOG_DEST = "log.txt"

# Log datetime
_C.LOG_TIME = ''

# ViDA parameters
_C.OPTIM.ViDALR = 5e-8
_C.TEST.vida_rank1 = 1
_C.TEST.vida_rank2 = 128
_C.OPTIM.MT_ViDA = 0.999
_C.OPTIM.MT = 0.999

# # Config destination (in SAVE_DIR)
# _C.CFG_DEST = "cfg.yaml"

_C.SETTING = "continual"
_C.ORDER = "default"

# --------------------------------- DPAL options ---------------------------- #
_C.DPAL = CfgNode()
_C.DPAL.NUM_PROMPT_TOKENS = 1
_C.DPAL.PROMPT_DEEP = True
_C.OPTIM.NORM_LR = 1e-2
_C.OPTIM.PROMPT_LR = 1e-2
_C.OPTIM.PREDICTOR_LR = 1e-2

_C.DPAL.ORTHO = 0.0
_C.DPAL.PROMPT_TYPE = "batch"

# --------------------------------- SVD options ---------------------------- #
_C.EXP_NAME = "test"

_C.SVD = CfgNode()
_C.SVD.ORDER = 1
_C.SVD.COMPONENT_NUM = 14
_C.SVD.MODE = 'svft'
_C.SVD.BIAS = False
_C.SVD.DECOMPOSE_QKV = False
_C.SVD.DYNAMIC = False
_C.SVD.NUM_SEEN_TEST_DOMAINS = 5
_C.SVD.CHECKPOINT = './dynamic_save'
_C.SVD.DYNAMIC_MODE = 'continual_adapt'
_C.SVD.TEMP = 1.0  # for combination mode
_C.SVD.WITHOUT_LASTBLOCKS = False
_C.SVD.FEATURE_LAYER = [-1]
_C.SVD.DYNAMIC_POOL = 'none'
_C.SVD.DSS_K = 5.0
_C.RECURRING = 1
_C.RECURRING_TYPE = 'same' # 'split'
_C.SVD.LOAD_OPTIMIZER_STATE = 'continue'
_C.SVD.MEASURE_FORGETTING = False

_C.SVD.VM_WEIGHT = 0.0
_C.SVD.VM_RATIO = 1.0
_C.SVD.VM_NUM = 1
# _C.SVD.DD_DISTANCE = 'std'
_C.SVD.DOMAIN_DISTANCE = 'kl_div'
_C.SVD.RETRIEVAL_DISTANCE = 'kl_div'

# --------------------------------- Source guidance options ---------------------------- #
_C.SOURCE_GUIDANCE = CfgNode()
_C.SOURCE_GUIDANCE.TEST_TYPE = 'source'
_C.SOURCE_GUIDANCE.N = 32
# --------------------------------- EATA options ---------------------------- #
_C.EATA = CfgNode()

# Fisher alpha. If set to 0.0, EATA becomes ETA and no EWC regularization is used
_C.EATA.FISHER_ALPHA = 2000.0

# Diversity margin
_C.EATA.D_MARGIN = 0.05
_C.EATA.MARGIN_E0 = 0.4             # Will be multiplied by: EATA.MARGIN_E0 * math.log(num_classes)

_C.EATA.USE_CONSISTENCY = False

# --------------------------------- SAR options ---------------------------- #
_C.SAR = CfgNode()

# Threshold e_m for model recovery scheme
_C.SAR.RESET_CONSTANT_EM = 0.2

# --------------------------------- DeYO options ---------------------------- #
_C.DEYO = CfgNode()
_C.DEYO.MARGIN_DEYO = 0.5
_C.DEYO.MARGIN_E0 = 0.4  # Will be multiplied by: DEYO.MARGIN_E0 * math.log(num_classes)
_C.DEYO.PATCH_LEN = 4
_C.DEYO.AUG_TYPE = 'patch'
_C.DEYO.OCCLUSION_SIZE = 112
_C.DEYO.ROW_START = 56
_C.DEYO.COLUMN_START = 56
_C.DEYO.PLPD_THRESHOLD = 0.3
_C.DEYO.FILTER_ENT = 1
_C.DEYO.FILTER_PLPD = 1
_C.DEYO.REWEIGHT_ENT = 1
_C.DEYO.REWEIGHT_PLPD = 1

# --------------------------------- Default config -------------------------- #


_CFG_DEFAULT = _C.clone()
_CFG_DEFAULT.freeze()


def assert_and_infer_cfg():
    """Checks config values invariants."""
    err_str = "Unknown adaptation method."
    assert _C.MODEL.ADAPTATION in ["source", "norm", "tent"]
    err_str = "Log destination '{}' not supported"
    assert _C.LOG_DEST in ["stdout", "file"], err_str.format(_C.LOG_DEST)


def merge_from_file(cfg_file):
    with g_pathmgr.open(cfg_file, "r") as f:
        cfg = _C.load_cfg(f)
    _C.merge_from_other_cfg(cfg)


def dump_cfg():
    """Dumps the config to the output directory."""
    cfg_file = os.path.join(_C.SAVE_DIR, _C.CFG_DEST)
    with g_pathmgr.open(cfg_file, "w") as f:
        _C.dump(stream=f)


def load_cfg(out_dir, cfg_dest="config.yaml"):
    """Loads config from specified output directory."""
    cfg_file = os.path.join(out_dir, cfg_dest)
    merge_from_file(cfg_file)


def reset_cfg():
    """Reset config to initial state."""
    cfg.merge_from_other_cfg(_CFG_DEFAULT)


def load_cfg_fom_args(description="Config options."):
    """Load config from command line args and set any specified options."""
    current_time = datetime.now().strftime("%y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--cfg", dest="cfg_file", type=str, required=True,
                        help="Config file location")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER,
                        help="See conf.py for all options")
    parser.add_argument("--data_dir", default="/mnt/nfs_shared_data/dataset/test_time_adaptation_data", type=str)
    parser.add_argument("--checkpoint", default=None, type=str)
    parser.add_argument("--unc_thr", default=0.2, type=float)
    parser.add_argument("--exp_name", default="test", type=str)
    parser.add_argument("--debug", action="store_true")
    # EATA Fisher Matrix
    parser.add_argument("--corruption", default="original", type=str)
    parser.add_argument("--level", default=5, type=int)

    
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()

    merge_from_file(args.cfg_file)
    cfg.merge_from_list(args.opts)

    cfg.EXP_NAME = args.exp_name
    cfg.DATA_DIR = args.data_dir
    cfg.TEST.ckpt = args.checkpoint

    log_dest = os.path.basename(args.cfg_file)
    log_dest = log_dest.replace('.yaml', '_{}.txt'.format(current_time))

    g_pathmgr.mkdirs(cfg.SAVE_DIR)
    cfg.LOG_TIME, cfg.LOG_DEST = current_time, log_dest

    if cfg.ORDER == "default":
        pass
    elif cfg.ORDER == "order0":
    #      brightness
    #   - pixelate
    #   - gaussian_noise
    #   - motion_blur
    #   - zoom_blur
    #   - glass_blur
    #   - impulse_noise
    #   - jpeg_compression
    #   - defocus_blur
    #   - elastic_transform
    #   - shot_noise
    #   - frost
    #   - snow
    #   - fog
    #   - contrast
        cfg.CORRUPTION.TYPE = ['brightness', 'pixelate', 'gaussian_noise', 'motion_blur', 'zoom_blur', 'glass_blur', 'impulse_noise', 'jpeg_compression', 'defocus_blur', 'elastic_transform', 'shot_noise', 'frost', 'snow', 'fog', 'contrast']
    elif cfg.ORDER == "order1":
    #   - jpeg_compression
    #   - shot_noise
    #   - zoom_blur
    #   - frost
    #   - contrast
    #   - fog
    #   - defocus_blur
    #   - elastic_transform
    #   - gaussian_noise
    #   - brightness
    #   - glass_blur
    #   - impulse_noise
    #   - pixelate
    #   - snow
    #   - motion_blur
        cfg.CORRUPTION.TYPE = ['jpeg_compression', 'shot_noise', 'zoom_blur', 'frost', 'contrast', 'fog', 'defocus_blur', 'elastic_transform', 'gaussian_noise', 'brightness', 'glass_blur', 'impulse_noise', 'pixelate', 'snow', 'motion_blur']
    elif cfg.ORDER == "order2":
    #   - contrast
    #   - defocus_blur
    #   - gaussian_noise
    #   - shot_noise
    #   - snow
    #   - frost
    #   - glass_blur
    #   - zoom_blur
    #   - elastic_transform
    #   - jpeg_compression
    #   - pixelate
    #   - brightness
    #   - impulse_noise
    #   - motion_blur
    #   - fog
        cfg.CORRUPTION.TYPE = ['contrast', 'defocus_blur', 'gaussian_noise', 'shot_noise', 'snow', 'frost', 'glass_blur', 'zoom_blur', 'elastic_transform', 'jpeg_compression', 'pixelate', 'brightness', 'impulse_noise', 'motion_blur', 'fog']
    elif cfg.ORDER == "order3":
    #   - shot_noise
    #   - fog
    #   - glass_blur
    #   - pixelate
    #   - snow
    #   - elastic_transform
    #   - brightness
    #   - impulse_noise
    #   - defocus_blur
    #   - frost
    #   - contrast
    #   - gaussian_noise
    #   - motion_blur
    #   - jpeg_compression
    #   - zoom_blur
        cfg.CORRUPTION.TYPE = ['shot_noise', 'fog', 'glass_blur', 'pixelate', 'snow', 'elastic_transform', 'brightness', 'impulse_noise', 'defocus_blur', 'frost', 'contrast', 'gaussian_noise', 'motion_blur', 'jpeg_compression', 'zoom_blur']
    elif cfg.ORDER == "order4":
    #     - pixelate
    #   - glass_blur
    #   - zoom_blur
    #   - snow
    #   - fog
    #   - impulse_noise
    #   - brightness
    #   - motion_blur
    #   - frost
    #   - jpeg_compression
    #   - gaussian_noise
    #   - shot_noise
    #   - contrast
    #   - defocus_blur
    #   - elastic_transform
        cfg.CORRUPTION.TYPE = ['pixelate', 'glass_blur', 'zoom_blur', 'snow', 'fog', 'impulse_noise', 'brightness', 'motion_blur', 'frost', 'jpeg_compression', 'gaussian_noise', 'shot_noise', 'contrast', 'defocus_blur', 'elastic_transform']
    elif cfg.ORDER == "order5":
    #   - motion_blur
    #   - snow
    #   - fog
    #   - shot_noise
    #   - defocus_blur
    #   - contrast
    #   - zoom_blur
    #   - brightness
    #   - frost
    #   - elastic_transform
    #   - glass_blur
    #   - gaussian_noise
    #   - pixelate
    #   - jpeg_compression
    #   - impulse_noise
        cfg.CORRUPTION.TYPE = ['motion_blur', 'snow', 'fog', 'shot_noise', 'defocus_blur', 'contrast', 'zoom_blur', 'brightness', 'frost', 'elastic_transform', 'glass_blur', 'gaussian_noise', 'pixelate', 'jpeg_compression', 'impulse_noise']
    elif cfg.ORDER == "order6":
    #   - frost
    #   - impulse_noise
    #   - jpeg_compression
    #   - contrast
    #   - zoom_blur
    #   - glass_blur
    #   - pixelate
    #   - snow
    #   - defocus_blur
    #   - motion_blur
    #   - brightness
    #   - elastic_transform
    #   - shot_noise
    #   - fog
    #   - gaussian_noise
        cfg.CORRUPTION.TYPE = ['frost', 'impulse_noise', 'jpeg_compression', 'contrast', 'zoom_blur', 'glass_blur', 'pixelate', 'snow', 'defocus_blur', 'motion_blur', 'brightness', 'elastic_transform', 'shot_noise', 'fog', 'gaussian_noise']
    elif cfg.ORDER == "order7":
    #   - glass_blur
    #   - zoom_blur
    #   - impulse_noise
    #   - fog
    #   - snow
    #   - jpeg_compression
    #   - gaussian_noise
    #   - frost
    #   - shot_noise
    #   - brightness
    #   - contrast
    #   - motion_blur
    #   - pixelate
    #   - defocus_blur
    #   - elastic_transform  
        cfg.CORRUPTION.TYPE = ['glass_blur', 'zoom_blur', 'impulse_noise', 'fog', 'snow', 'jpeg_compression', 'gaussian_noise', 'frost', 'shot_noise', 'brightness', 'contrast', 'motion_blur', 'pixelate', 'defocus_blur', 'elastic_transform']
    elif cfg.ORDER == "order8":
    #   - defocus_blur
    #   - motion_blur
    #   - zoom_blur
    #   - shot_noise
    #   - gaussian_noise
    #   - glass_blur
    #   - jpeg_compression
    #   - fog
    #   - contrast
    #   - pixelate
    #   - frost
    #   - snow
    #   - brightness
    #   - elastic_transform
    #   - impulse_noise 
        cfg.CORRUPTION.TYPE = ['defocus_blur', 'motion_blur', 'zoom_blur', 'shot_noise', 'gaussian_noise', 'glass_blur', 'jpeg_compression', 'fog', 'contrast', 'pixelate', 'frost', 'snow', 'brightness', 'elastic_transform', 'impulse_noise']
    elif cfg.ORDER == "order9":
    #   - contrast
    #   - gaussian_noise
    #   - defocus_blur
    #   - zoom_blur
    #   - frost
    #   - glass_blur
    #   - jpeg_compression
    #   - fog
    #   - pixelate
    #   - elastic_transform
    #   - shot_noise
    #   - impulse_noise
    #   - snow
    #   - motion_blur
    #   - brightness
        cfg.CORRUPTION.TYPE = ['contrast', 'gaussian_noise', 'defocus_blur', 'zoom_blur', 'frost', 'glass_blur', 'jpeg_compression', 'fog', 'pixelate', 'elastic_transform', 'shot_noise', 'impulse_noise', 'snow', 'motion_blur', 'brightness']
    elif cfg.ORDER == "test":
        cfg.CORRUPTION.TYPE = ['contrast', 'impulse_noise', 'snow', 'motion_blur', 'pixelate']
    else:
        raise ValueError("Unknown order")
    cfg.freeze()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(filename)s: %(lineno)4d]: %(message)s",
        datefmt="%y/%m/%d %H:%M:%S",
        handlers=[
            logging.FileHandler(os.path.join(cfg.SAVE_DIR, cfg.LOG_DEST)),
            logging.StreamHandler()
        ])

    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    random.seed(cfg.RNG_SEED)
    torch.backends.cudnn.benchmark = cfg.CUDNN.BENCHMARK

    logger = logging.getLogger(__name__)
    version = [torch.__version__, torch.version.cuda,
               torch.backends.cudnn.version()]
    logger.info(
        "PyTorch Version: torch={}, cuda={}, cudnn={}".format(*version))
    logger.info(cfg)
    return args