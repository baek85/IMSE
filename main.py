import logging

import torch
import torch.optim as optim

from robustbench.data import load_imagenetc
from robustbench.model_zoo.enums import ThreatModel
from robustbench.utils import load_model
from robustbench.utils import clean_accuracy as accuracy

from method import tent, cotta, vida, svd
import math

import wandb
from conf import cfg, load_cfg_fom_args
from selectedRotateImageFolder import prepare_test_data
import os
from utils import plot_domain_change_detection, evaluate_domain_change_detection
logger = logging.getLogger(__name__)

def evaluate(description):
    args = load_cfg_fom_args(description)
    exp_name = "{}_{}_{}".format(cfg.MODEL.ARCH, cfg.MODEL.ADAPTATION, args.exp_name)
    if not args.debug:
        wandb.init(project="IMSE", name=exp_name, config=cfg)

    base_model = load_model(cfg.MODEL.ARCH, cfg.CKPT_DIR,
                       cfg.CORRUPTION.DATASET, ThreatModel.corruptions).cuda()
    
    if cfg.MODEL.ADAPTATION == "source":
        logger.info("test-time adaptation: NONE")
        model = setup_source(base_model)
    elif cfg.MODEL.ADAPTATION == "tent":
        logger.info("test-time adaptation: TENT")
        model = setup_tent(base_model)
    elif cfg.MODEL.ADAPTATION == "cotta":
        logger.info("test-time adaptation: CoTTA")
        model = setup_cotta(base_model)
    elif cfg.MODEL.ADAPTATION == "vida":
        logger.info("test-time adaptation: ViDA")
        model = setup_vida(args, base_model)
    elif cfg.MODEL.ADAPTATION == "imse":
        logger.info("test-time adaptation: IMSE")
        model = setup_svd(base_model)
    else:
        raise NotImplementedError("test-time adaptation: %s" % cfg.MODEL.ADAPTATION)

    torch.cuda.empty_cache()
    All_acc = []
    domain_list = []

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Number of trainable parameters: {num_params}")
    actual_changes = []
    vertical_lines = []


    if cfg.RECURRING_TYPE == "split":
        split_size = cfg.CORRUPTION.NUM_EX

        split_index_list = []
        indices = torch.randperm(50000)
        for i in range(cfg.RECURRING):
            start_idx = i * split_size
            end_idx = start_idx + split_size
            split_index_list.append(indices[start_idx:end_idx])


    for recur_idx in range(cfg.RECURRING):
        mean_acc_list = []
        num_domains = len(cfg.CORRUPTION.TYPE) * len(cfg.CORRUPTION.SEVERITY)
        domain_name_list = []
        for i_x, corruption_type in enumerate(cfg.CORRUPTION.TYPE):
            for ii, severity in enumerate(cfg.CORRUPTION.SEVERITY):
                cur_idx = ii + i_x * len(cfg.CORRUPTION.SEVERITY) + recur_idx * len(cfg.CORRUPTION.TYPE) * len(cfg.CORRUPTION.SEVERITY)
                domain_list.append('{}_{}'.format(corruption_type, severity))
                domain_name_list.append(corruption_type)
                if cur_idx == 0:
                    try:
                        model.reset()
                    except:
                        pass
                    logger.info("resetting model")
                elif cfg.SETTING == "reset_each_shift":
                    if cur_idx > 0 and cfg.SVD.DYNAMIC and cfg.SETTING == "reset_each_shift":
                        model.domain_shift_detected()
                        logger.info("oracle domain shift")
                        model.reset_soft()
                    else:
                        model.reset()
                        logger.info("resetting model")
                elif cur_idx > 0 and cfg.SVD.DYNAMIC and cfg.SETTING in ["continual"]:
                    pass
                else:
                    logger.warning("not resetting model")

                if cfg.RECURRING_TYPE == "split":
                    split_index = split_index_list[recur_idx]
                    x_test, y_test = load_imagenetc(50000,
                                            severity, cfg.DATA_DIR, False,
                                            [corruption_type],
                                            prepr='Res256Crop224',
                                            indices=split_index)
                else:
                    x_test, y_test = load_imagenetc(cfg.CORRUPTION.NUM_EX,
                                                severity, cfg.DATA_DIR, False,
                                                [corruption_type],
                                                prepr='Res256Crop224')
                                                
                acc = accuracy(model, x_test, y_test, cfg.TEST.BATCH_SIZE, debug=args.debug)

                len_test_loader = math.ceil(len(y_test) / cfg.TEST.BATCH_SIZE)
                actual_changes.append(cur_idx * len_test_loader)
                if cfg.SVD.DYNAMIC_POOL in ['dss']:
                    vertical_lines.append(len(model.dss_list))
                All_acc.append(acc)
                logger.info(f"{cur_idx}th_acc % [{corruption_type}{severity}]: {acc:.2%}")

                if not args.debug:
                    wandb.log({f"{cur_idx}th_acc_{corruption_type}{severity}": acc, "acc_list": acc})
                torch.cuda.empty_cache()

                mean_acc_list.append(acc)

        if len(mean_acc_list) > 0:
            recurring_mean_acc = sum(mean_acc_list) / len(mean_acc_list)
            logger.info(f"{recur_idx}th acc % [mean]: {recurring_mean_acc:.2%}")
            if not args.debug:
                wandb.log({f"{recur_idx}th_mean_acc": recurring_mean_acc,
                            "mean_acc": recurring_mean_acc})

    if cfg.SVD.NUM_SEEN_TEST_DOMAINS == 15 and cfg.SVD.DYNAMIC:
        model.domain_shift_detected()
        
    logger.info(f"total_acc % [mean]: {sum(All_acc)/len(All_acc):.2%}")
    if not args.debug:
        wandb.log({"total_mean_acc": sum(All_acc)/len(All_acc)})
    
    if cfg.SVD.DYNAMIC_POOL in ['dss']:
        predicted_changes = model.predicted_changes
        logger.info(f"actual_changes: {actual_changes}")
        logger.info(f"predicted_changes: {predicted_changes}")
        file_path = 'logs/domain_change_detection_dss/{}/'.format(exp_name)
        file_path += '{}/{}_K_{}'.format(cfg.ORDER, cfg.SVD.FEATURE_LAYER, cfg.SVD.DSS_K)
        eval_file_path = None
        
        if not os.path.exists(os.path.dirname(file_path)):
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

        dss_accuracy, precision, recall, f1 = evaluate_domain_change_detection(cfg, actual_changes, predicted_changes, eval_file_path, iterations=num_domains * len_test_loader)
        if not args.debug:
            wandb.log({"dss_accuracy": dss_accuracy, "dss_precision": precision, "dss_recall": recall, "dss_f1": f1})
        
        logger.info(f"dss_accuracy: {dss_accuracy}, precision: {precision}, recall: {recall}, f1: {f1}")
        plot_domain_change_detection(model, cfg, vertical_lines, file_path)
            
        wandb.log({"domain_change_detection": wandb.Image(file_path+'_DSS.png', caption=exp_name)})

    return

def setup_source(model):
    """Set up the baseline source model without adaptation."""
    model.eval()
    logger.info(f"model for evaluation: %s", model)
    return model

def setup_tent(model):
    """Set up tent adaptation.

    Configure the model for training + feature modulation by batch statistics,
    collect the parameters for feature modulation by gradient optimization,
    set up the optimizer, and then tent the model.
    """
    model = tent.configure_model(model)
    params, param_names = tent.collect_params(model)
    optimizer = setup_optimizer(params)
    tent_model = tent.Tent(model, optimizer,
                           steps=cfg.OPTIM.STEPS,
                           episodic=cfg.MODEL.EPISODIC)
    logger.info(f"model for adaptation: %s", model)
    logger.info(f"params for adaptation: %s", param_names)
    logger.info(f"optimizer for adaptation: %s", optimizer)
    return tent_model

def setup_optimizer(params):
    """Set up optimizer for tent adaptation.

    Tent needs an optimizer for test-time entropy minimization.
    In principle, tent could make use of any gradient optimizer.
    In practice, we advise choosing Adam or SGD+momentum.
    For optimization settings, we advise to use the settings from the end of
    trainig, if known, or start with a low learning rate (like 0.001) if not.

    For best results, try tuning the learning rate and batch size.
    """
    if cfg.OPTIM.METHOD == 'Adam':
        return optim.Adam(params,
                    lr=cfg.OPTIM.LR,
                    betas=(cfg.OPTIM.BETA, 0.999),
                    weight_decay=cfg.OPTIM.WD)
    elif cfg.OPTIM.METHOD == 'SGD':
        return optim.SGD(params,
                   lr=cfg.OPTIM.LR,
                   momentum=0.9,
                   dampening=0,
                   weight_decay=cfg.OPTIM.WD,
                   nesterov=True)
    else:
        raise NotImplementedError

def setup_cotta(model):
    """Set up tent adaptation.

    Configure the model for training + feature modulation by batch statistics,
    collect the parameters for feature modulation by gradient optimization,
    set up the optimizer, and then tent the model.
    """
    model = cotta.configure_model(model)
    params, param_names = cotta.collect_params(model)
    optimizer = setup_optimizer(params)
    cotta_model = cotta.CoTTA(model, optimizer,
                           steps=cfg.OPTIM.STEPS,
                           episodic=cfg.MODEL.EPISODIC)
    logger.info(f"model for adaptation: %s", model)
    logger.info(f"params for adaptation: %s", param_names)
    logger.info(f"optimizer for adaptation: %s", optimizer)
    return cotta_model

def setup_vida(args, model):
    model = vida.configure_model(model, cfg)
    model_param, vida_param = vida.collect_params(model)
    optimizer = setup_optimizer_vida(model_param, vida_param, cfg.OPTIM.LR, cfg.OPTIM.ViDALR)
    vida_model = vida.ViDA(model, optimizer,
                           steps=cfg.OPTIM.STEPS,
                           episodic=cfg.MODEL.EPISODIC,
                           unc_thr = args.unc_thr,
                           ema = cfg.OPTIM.MT,
                           ema_vida = cfg.OPTIM.MT_ViDA,
                           )
    logger.info(f"model for adaptation: %s", model)
    logger.info(f"optimizer for adaptation: %s", optimizer)
    return vida_model


def setup_svd(model):
    model = svd.configure_model(model, cfg)
    svd_params, svd_names = svd.collect_params(model, cfg)
    logger.info(f"params for adaptation: %s", svd_names)
    optimizer = svd.get_optimizer(cfg, svd_params)

    source_domain_dataset, source_domain_loader = prepare_test_data(data_dir='/mnt/nfs_shared_data/dataset/ILSVRC2012', use_transforms=True, batch_size=64, if_shuffle=False, num_workers=0)
    source_domain_dataset.set_dataset_size(2000)
    source_domain_dataset.switch_mode(True, False)

    svd_model = svd.SVD(cfg, model, optimizer, scheduler=None, params=svd_params, source_domain_loader=source_domain_loader)
    return svd_model

def setup_optimizer_vida(params, params_vida, model_lr, vida_lr):
    if cfg.OPTIM.METHOD == 'Adam':
        return optim.Adam([{"params": params, "lr": model_lr},
                                  {"params": params_vida, "lr": vida_lr}],
                                 lr=1e-5, betas=(cfg.OPTIM.BETA, 0.999),weight_decay=cfg.OPTIM.WD)

    elif cfg.OPTIM.METHOD == 'SGD':
        return optim.SGD([{"params": params, "lr": model_lr},
                                  {"params": params_vida, "lr": vida_lr}],
                                    momentum=cfg.OPTIM.MOMENTUM,dampening=cfg.OPTIM.DAMPENING,
                                    nesterov=cfg.OPTIM.NESTEROV,
                                 lr=1e-5,weight_decay=cfg.OPTIM.WD)
    else:
        raise NotImplementedError

def setup_optimizer_svd(params, lr):
    if cfg.OPTIM.METHOD == 'Adam':
        return optim.Adam([{"params": params, "lr": lr}],
                                 lr=1e-5, betas=(cfg.OPTIM.BETA, 0.999),weight_decay=cfg.OPTIM.WD)

    elif cfg.OPTIM.METHOD == 'SGD':
        return optim.SGD([{"params": params, "lr": lr}],
                                    momentum=cfg.OPTIM.MOMENTUM,dampening=cfg.OPTIM.DAMPENING,
                                    nesterov=cfg.OPTIM.NESTEROV,
                                 lr=1e-5,weight_decay=cfg.OPTIM.WD)
    else:
        raise NotImplementedError

if __name__ == '__main__':
    evaluate('"Imagenet-C evaluation.')
