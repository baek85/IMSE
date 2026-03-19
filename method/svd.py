from copy import deepcopy
import torch
import torch.nn as nn
from .decompose_svd import convert_linear_to_svdlinear, SVDLinear
import logging
import math
from utils import get_imagenet_r_mask
import torch.nn.functional as F
import torch.optim as optim
from .sam import SAM
from .domain_info import DomainInfo
from tqdm import tqdm
imagenet_r_mask = get_imagenet_r_mask()
logger = logging.getLogger(__name__)

mode_dict = {
            'continual_no_adapt': 'no_adapt',
            'continual_adapt': 'adapt',
            'mean_no_adapt': 'no_adapt',
            'mean_adapt': 'adapt',
            'top1_no_adapt': 'no_adapt',
            'top1_adapt': 'adapt',
            'mix_no_adapt': 'no_adapt',
            'mix_adapt': 'adapt',
        }

vit_component = [['attn.proj'], # 0
            ['attn.qkv'], # 1
            ['mlp.fc1'], # 2
            ['mlp.fc2'], # 3
            ['attn.proj','attn.qkv'], # 4
            ['mlp.fc1','mlp.fc2'], # 5
            ['attn.proj','mlp.fc1'], # 6
            ['attn.proj','mlp.fc2'], # 7
            ['attn.qkv','mlp.fc1'], # 8
            ['attn.qkv','mlp.fc2'], # 9
            ['attn.proj','attn.qkv','mlp.fc1'], # 10
            ['attn.proj','attn.qkv','mlp.fc2'], # 11
            ['attn.proj','mlp.fc1','mlp.fc2'], # 12
            ['attn.qkv','mlp.fc1','mlp.fc2'], # 13
            ['attn.proj','attn.qkv','mlp.fc1','mlp.fc2']] # 14

class SVD(DomainInfo):
    """SAR online adapts a model by Sharpness-Aware and Reliable entropy minimization during testing.
    Once SARed, a model adapts itself by updating on every forward.
    """
    def __init__(self, cfg, model, optimizer, scheduler, params, source_domain_loader, steps=1, episodic=False, num_classes=1000):
        self.cfg = cfg
        if self.cfg.SVD.DYNAMIC_POOL in ['dss']:
            frozen_model = deepcopy(model)
            for param in frozen_model.parameters():
                param.detach_()
            super().__init__(cfg, frozen_model)
            del self.frozen_model
        else:
            super().__init__(cfg, None)
        
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.source_domain_loader = source_domain_loader
        self.steps = steps
        assert steps > 0, "SAR requires >= 1 step(s) to forward and update"
        self.episodic = episodic

        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)

        self.encountered_test_domains = 0
        self.saved_test_domains = 0

        self.no_adapt = False
        self.method = cfg.MODEL.ADAPTATION

        self.vm_weight = cfg.SVD.VM_WEIGHT
        self.svd_mode = cfg.SVD.MODE
        self.vm_ratio = cfg.SVD.VM_RATIO
        self.vm_num = cfg.SVD.VM_NUM
        
        self.num_classes = num_classes
        self.init_svd()
        self.softmax_entropy = softmax_entropy
        
        self.forward_mode = 'adapt'
        self.dynamic_mode = 'default_adapt'

        self.is_dynamic = self.cfg.SVD.DYNAMIC
        self.feature_layers = getattr(cfg.SVD, 'FEATURE_LAYER')
        
        logger.info(f"self.feature_layers: {self.feature_layers}")
        logger.info(f"Extract Source Domain Information")
        with torch.no_grad():
            for iter_, (images, targets) in tqdm(enumerate(self.source_domain_loader, start=1)):      
                super().forward(images.cuda(), update=False, no_dss=True)
                if iter_ == len(self.source_domain_loader):
                        break
        self.save_domain_info(save_singular_value=False, save_optimizer_state=False)
        logger.info(f"Save Source Domain Information")
        self.domain_optimizer_states = []
        self.domain_optimizer_states.append(self.optimizer.state_dict())
        logger.info(f"Save Source Optimizer State")

        self.original_optimizer_state = self.optimizer.state_dict()
            
        self.encountered_test_domains += 1
        self.saved_test_domains += 1

        self.domain_training_counts = {}
        self.domain_training_counts[0] = 0
        self.domain_training_counts[1] = 1
    
    @torch.no_grad()
    def forward_embedding_patch_embed(self, x):
        features = {}
        if self.cfg.MODEL.ARCH == 'VITB_augreg_in21k_ours':
            x = self.model.normalize(x)
            x = self.model.model.patch_embed(x)
            x = self.model.model._pos_embed(x)
            x = self.model.model.norm_pre(x)
            features[-1] = x[:, 1:].clone()
        else:
            x = self.model.patch_embed(x)
            x = self.model._pos_embed(x)
            x = self.model.norm_pre(x)
            features[-1] = x[:, 1:].clone()
            
        return features

    def save_domain_info(self, save_singular_value=True, save_optimizer_state=True):
        """Save the model and optimizer states."""
        if save_singular_value:
            for name, module in self.model.named_modules():
                if isinstance(module, SVDLinear):
                    module.save_singular_value()
            logger.info("save singular value")

        for layer in self.feature_layers:
            self.domain_infos[layer].append((self.domain_var[layer], self.domain_mean[layer]))
        
        if save_optimizer_state:
            self.domain_optimizer_states.append(self.optimizer.state_dict())


        self.reset_domain_info()

    def get_num_prev_domains(self):
        return len(self.domain_training_counts)
    
    def init_svd(self):
        self.margin_e0 = self.cfg.EATA.MARGIN_E0 *math.log(self.num_classes)

    def forward(self, x, no_adapt=False):
        if self.episodic:
            self.reset()

        if no_adapt:
            return self.forward_no_adapt(x)

        for _ in range(self.steps):

            if self.cfg.SVD.DYNAMIC_POOL in ['dss']:
                super().forward(x, update=False)
                if self.num_step == 1 and self.encountered_test_domains > self.cfg.SVD.NUM_SEEN_TEST_DOMAINS and self.is_dynamic:
                    if len(self.feature_layers) == 1:
                        layer = self.feature_layers[0]
                        try:
                            selected_domain = self.domain_shift_value_list_dict[layer].argmin()
                        except:
                            breakpoint()
                        logger.info(f"domain shift detected, now adapt {self.cfg.SVD.DYNAMIC_MODE} mode")
                        self.global_selected_domain = selected_domain.item()
                        self.global_domain_shift_value_list = self.domain_shift_value_list_dict[layer]
                        self.setting_dynamic_mode(idx=selected_domain.item(), domain_shift_value_list=self.domain_shift_value_list_dict[layer], data=x)
                    else:
                        raise ValueError(f"Length of feature layers is not 1, {len(self.feature_layers)}")
            else:
                pass

            if self.forward_mode == 'no_adapt':
                outputs, loss_dict = self.forward_and_no_adapt(x)
            elif self.forward_mode == 'adapt':
                outputs, reset_flag, loss_dict = self.forward_and_adapt_sar(x)
                if self.scheduler is not None:
                    self.scheduler.step()

            if hasattr(self, 'domain_shift_value'):
                loss_dict['domain_shift_value'] = self.domain_shift_value
            if hasattr(self, 'relative_domain_shift_value'):
                loss_dict['relative_domain_shift_value'] = self.relative_domain_shift_value


            loss_dict['optimze_times'] = self.domain_training_counts[self.get_num_prev_domains()-1]

        return outputs, loss_dict
    

    @torch.no_grad()
    def forward_no_adapt(self, x):
        """Forward the model without adaptation.
        This is used for testing the model without DPAL.
        """
        outputs = self.model(x)
        return outputs
    
    def reset_model_probs(self, probs):
        self.current_model_probs = probs
    
    @torch.no_grad()
    def forward_and_no_adapt(self, x):
        """Forward model input data.
        Measure entropy of the model prediction, but do not take gradients or update params.
        """
        outputs = self.model(x)
        
        entropys = self.softmax_entropy(outputs)
        filter_ids_1 = torch.where(entropys < self.margin_e0)
        entropys = entropys[filter_ids_1]
        loss_dict = {}
        loss_dict['reliable_samples'] = entropys.size(0)
        loss_dict['ce_loss'] = entropys.mean(0).item()
        
        return outputs, loss_dict

    
    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt_sar(self, x):
        """Forward and adapt model input data.
        Measure entropy of the model prediction, take gradients, and update params.
        """
        self.optimizer.zero_grad()

        outputs = self.model(x)
        entropys = self.softmax_entropy(outputs)
        filter_ids_1 = torch.where(entropys < self.margin_e0)
        entropys = entropys[filter_ids_1]
        loss = entropys.mean(0)
                    
        if self.svd_mode in ['svft_variance_maximization'] and self.vm_weight > 0.0:
            vm_loss = self.get_variance_maximization_loss()
            loss -= self.vm_weight * vm_loss
        loss.backward()

        self.optimizer.first_step(zero_grad=True)
        
        entropys2 = self.softmax_entropy(self.model(x))
        entropys2 = entropys2[filter_ids_1]  # second time forward  
        filter_ids_2 = torch.where(entropys2 < self.margin_e0)  # here filtering reliable samples again, since model weights have been changed to \Theta+\hat{\epsilon(\Theta)}
        loss_second = entropys2[filter_ids_2].mean(0)
        loss_dict = {}


        loss_dict['reliable_samples'] = entropys2[filter_ids_2].size(0)
        loss_dict['ce_loss'] = loss_second.item()
        if self.svd_mode in ['svft_variance_maximization']:
            vm_loss = self.get_variance_maximization_loss()
            if self.vm_weight > 0.0:
                loss_second -= self.vm_weight * vm_loss
            loss_dict['vm_loss'] = vm_loss.item()

        loss_second.backward()
        self.optimizer.second_step(zero_grad=True)
        reset_flag = False

        return outputs, reset_flag, loss_dict

    def get_variance_maximization_loss(self):
        loss = 0.
        for name, module in self.model.named_modules():
            if isinstance(module, SVDLinear):
                current_loss = module.variance_maximization(self.vm_ratio)
                if current_loss is not None:
                    loss += current_loss
            
        loss /= self.vm_num
        if loss == 0.0:
            loss = torch.tensor(0.0)
        return loss

    def reset_soft(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state, strict=False)
        self.coreset = []
        self.init_svd()
        logger.info(f"soft reset model")

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state, strict=True)
        self.ema = None
        self.init_svd()
        logger.info(f"reset model")
    
    def save_singular_value(self):
        """Save the model and optimizer states."""
        for name, module in self.model.named_modules():
            if isinstance(module, SVDLinear):
                module.save_singular_value()
    
    def save(self):
        self.save_singular_value()
        self.save_domain_info(save_singular_value=False)
        logger.info(f"save singular value and domain info")
        # self.reset_domain_info()

    def domain_shift_detected(self, pass_test_domains=False):
        """Check if domain shift is detected."""
        self.num_step = 0

        if self.cfg.SVD.DYNAMIC_POOL in ['dss']:
            self.encountered_test_domains += 1
            self.save()
            self.saved_test_domains += 1
            logger.info(f"domain shift detected, save expert")
            logger.info(f"encountered num_experts: {self.encountered_test_domains}")
        
            
    def setting_dynamic_mode(self, idx=0, domain_shift_value_list=None, data=None):
        self.forward_mode = mode_dict[self.cfg.SVD.DYNAMIC_MODE]

        if self.encountered_test_domains < self.cfg.SVD.NUM_SEEN_TEST_DOMAINS:
            pass
        elif self.encountered_test_domains >= self.cfg.SVD.NUM_SEEN_TEST_DOMAINS:
            logger.info("domain shift value list: {}".format(domain_shift_value_list))
            logger.info("selected domain: {}".format(idx))
            self.domain_training_counts[self.get_num_prev_domains()] = self.domain_training_counts[idx] + 1           
            for name, module in self.model.named_modules():
                if isinstance(module, SVDLinear):
                    module.set_forward_mode(self.cfg.SVD.DYNAMIC_MODE, idx=idx, domain_shift_value_list=domain_shift_value_list)

            if self.forward_mode == 'no_adapt':
                pass
            elif self.forward_mode == 'adapt':               
                if self.cfg.SVD.LOAD_OPTIMIZER_STATE == 'load':
                    self.optimizer.load_state_dict(self.domain_optimizer_states[idx])
                    logger.info(f"Load optimizer state of {idx}th domain")
                elif self.cfg.SVD.LOAD_OPTIMIZER_STATE == 'reset':
                    self.optimizer.load_state_dict(self.original_optimizer_state)
                    logger.info(f"Reset optimizer state")
                elif self.cfg.SVD.LOAD_OPTIMIZER_STATE == 'continue':
                    logger.info(f"Continue optimizer state")
                    pass
                else:
                    raise ValueError(f"Unknown load optimizer state: {self.cfg.SVD.LOAD_OPTIMIZER_STATE}")

                if self.domain_training_counts[self.get_num_prev_domains()-1] >= 2:
                    logger.info(f"This parameter optimizes {self.domain_training_counts[self.get_num_prev_domains()-1]} times")
                elif self.domain_training_counts[self.get_num_prev_domains()-1] == 1:
                    logger.info(f"This parameter optimizes first time")

            self.dynamic_mode = self.cfg.SVD.DYNAMIC_MODE 

        
    def set_new_optimizer(self, lr_decay=1.0):
        params_list, names_list = collect_params(self.model, self.cfg, lr_decay=lr_decay)
        new_optimizer = get_optimizer(self.cfg, params_list)
        self.optimizer = new_optimizer

def get_optimizer(cfg, params_list):
    if cfg.OPTIM.METHOD == 'SAM_Adam':
        base_optimizer = torch.optim.Adam
        optimizer = SAM(params_list, base_optimizer)
    elif cfg.OPTIM.METHOD == 'SAM_SGD':
        base_optimizer = torch.optim.SGD
        optimizer = SAM(params_list, base_optimizer, momentum=0.9)
    elif cfg.OPTIM.METHOD == 'Adam':
        optimizer = optim.Adam(params_list,
                    betas=(cfg.OPTIM.BETA, 0.999),
                    weight_decay=cfg.OPTIM.WD)
    elif cfg.OPTIM.METHOD == 'SGD':
        optimizer = optim.SGD(params_list,
                   momentum=0.9,
                   dampening=0,
                   weight_decay=cfg.OPTIM.WD,
                   nesterov=True)
    else:
        raise NotImplementedError(f"Unknown optimization method: {cfg.OPTIM.METHOD}")
    
    return optimizer

# @torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

def collect_params(model, cfg, lr_decay=1.0):
    """Collect all trainable parameters.

    Walk the model's modules and collect all parameters.
    Return the parameters and their names.

    Note: other choices of parameterization are possible!
    """
    params_list = []
    names_list = []
    target_component_list = vit_component[cfg.SVD.COMPONENT_NUM]
    print(f"target_component_list: {target_component_list}")

    for nm, param in model.named_parameters():
        if cfg.MODEL.ADAPTATION in ['imse']:
            if cfg.MODEL.ARCH in ['VITB_augreg_in21k_ours', 'VITB_augreg_in21k_vida']:
                if 'blocks.9' in nm:
                    continue
                if 'blocks.10' in nm:
                    continue
                if 'blocks.11' in nm:
                    continue
        for target_component in target_component_list:
            if target_component in nm:
                if param.requires_grad:
                    params_list += [{'params': param, 'lr': cfg.OPTIM.LR * lr_decay}]
                    names_list.append(nm)
                    param.requires_grad = True

        if cfg.SVD.BIAS:
            if 'svft_bias' in nm:
                for target_component in target_component_list:
                    if target_component in nm:
                        params_list += [{'params': param, 'lr': cfg.OPTIM.LR * lr_decay}]
                        names_list.append(nm)
                        param.requires_grad = True
        
    if cfg.OPTIM.NORM_LR > 0.0:
        for nm, m in model.named_modules():
            if cfg.MODEL.ADAPTATION in ['imse']:
                if cfg.MODEL.ARCH in ['VITB_augreg_in21k_ours', 'VITB_augreg_in21k_vida']:
                    if 'blocks.9' in nm:
                        continue
                    if 'blocks.10' in nm:
                        continue
                    if 'blocks.11' in nm:
                        continue
                    if 'norm.' in nm:
                        continue
                    if nm in ['norm']:
                        continue

            if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
                for np, p in m.named_parameters():
                    if np in ['weight', 'bias']:
                        p.requires_grad_(True)
                        params_list += [{'params': p, 'lr': cfg.OPTIM.NORM_LR * lr_decay}]
                        names_list.append(f"{nm}.{np}")
                                    
    return params_list, names_list

def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state

def load_model_and_optimizer(model, optimizer, model_state, optimizer_state, strict=True):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=strict)
    optimizer.load_state_dict(optimizer_state)

def configure_model(model, cfg, component_num=0):
    """Configure model for use with tent."""
    # train mode, because tent optimizes the model to minimize entropy
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.cpu()
    model.requires_grad_(False)
    if cfg.SVD.DECOMPOSE_QKV:
        from timm.models.vision_transformer import Attention as OldAttention

        for i, blk in enumerate(model.model.blocks):
            if isinstance(blk.attn, OldAttention):
                old_attn = blk.attn
                new_attn = Attention(
                    dim=old_attn.qkv.in_features,
                    num_heads=old_attn.num_heads,
                    qkv_bias=old_attn.qkv.bias is not None,
                    qk_norm=isinstance(old_attn.q_norm, nn.LayerNorm),
                    proj_bias=old_attn.proj.bias is not None,
                    attn_drop=old_attn.attn_drop.p,
                    proj_drop=old_attn.proj_drop.p,
                    norm_layer=type(old_attn.q_norm)
                )
                convert_pretrained_qkv_to_split(old_attn, new_attn)
                new_attn.proj.load_state_dict(old_attn.proj.state_dict())
                blk.attn = new_attn
                    

    if cfg.SVD.MODE in ['svft_variance_maximization']:
        vm_block_list = []
        block_names = []
        for i, blk in enumerate(model.model.blocks):
            block_names.append(f"blocks.{i}")


        for block_name in block_names[-cfg.SVD.VM_NUM:]:
            vm_block_list.append(block_name)
        logger.info(f"VM block list: {vm_block_list}")
    else:
        vm_block_list = []

    svd_params, svd_names = convert_linear_to_svdlinear(model = model, target_replace_module = ["Attention", "Mlp"], 
                                                    mode=cfg.SVD.MODE,cfg=cfg, vm_block_list=vm_block_list, 
                                                    vm_weight=cfg.SVD.VM_WEIGHT)

    model.to(device)
    model.train()
    return model


def check_model(model):
    """Check model for compatability with tent."""
    is_training = model.training
    assert is_training, "tent needs train mode: call model.train()"
    param_grads = [p.requires_grad for p in model.parameters()]
    has_any_params = any(param_grads)
    has_all_params = all(param_grads)
    assert has_any_params, "tent needs params to update: " \
                           "check which require grad"
    assert not has_all_params, "tent should not update all params: " \
                               "check which require grad"
    has_bn = any([isinstance(m, nn.BatchNorm2d) for m in model.modules()])
    assert has_bn, "tent needs normalization for its optimization"

def update_model_probs(current_model_probs, new_probs):
    if current_model_probs is None:
        if new_probs.size(0) == 0:
            return None
        else:
            with torch.no_grad():
                return new_probs.mean(0).detach()
    else:
        if new_probs.size(0) == 0:
            with torch.no_grad():
                return current_model_probs
        else:
            with torch.no_grad():
                return 0.9 * current_model_probs + (1 - 0.9) * new_probs.mean(0).detach()

from typing import Any, Callable, Dict, Optional, Set, Tuple, Type, Union, List
from torch.jit import Final
from timm.layers import use_fused_attn
import torch.nn as nn

class Attention(nn.Module):
    fused_attn: Final[bool]

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.,
        proj_drop: float = 0.,
        norm_layer: [nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        # 3-way split to mimic original qkv
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)

        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape

        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

def convert_pretrained_qkv_to_split(attn_old: nn.Module, attn_new: nn.Module):
    qkv_weight = attn_old.qkv.weight  # shape: [3*dim, dim]
    qkv_bias = attn_old.qkv.bias      # shape: [3*dim]

    dim = qkv_weight.shape[1]
    attn_new.q.weight.data.copy_(qkv_weight[:dim])
    attn_new.k.weight.data.copy_(qkv_weight[dim:2*dim])
    attn_new.v.weight.data.copy_(qkv_weight[2*dim:])

    if qkv_bias is not None:
        attn_new.q.bias.data.copy_(qkv_bias[:dim])
        attn_new.k.bias.data.copy_(qkv_bias[dim:2*dim])
        attn_new.v.bias.data.copy_(qkv_bias[2*dim:])