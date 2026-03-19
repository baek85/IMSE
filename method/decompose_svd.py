import math
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import PIL
import torch
import torch.nn.functional as F

import torch.nn as nn
import logging

logger = logging.getLogger(__name__)
    

def get_svd(svft_weight, type='weight'):
    svft_weight = svft_weight.cuda()
    U, S, Vh = torch.linalg.svd(svft_weight, full_matrices=True)
    r = torch.sum(S > 1e-6)

    U_subspace = U[:, :r]
    S_subspace = S[:r]
    Vh_subspace = Vh[:r, :]
    if type == 'weight':
        error = svft_weight - U_subspace @ torch.diag(S_subspace) @ Vh_subspace

    if U.size(0) - r > 0:
        U_null = U[:, r:]
    else:
        U_null = None
    
    if Vh.size(1) - r > 0:
        V_null = Vh[r:, :]
    else:
        V_null = None

    if U.size(0) - r > 0 or Vh.size(1) - r > 0:
        S_null = torch.zeros(U.size(0) - r + Vh.size(1) - r, device=S.device, dtype=S.dtype)
    else:
        S_null = None
    return U_subspace, S_subspace, Vh_subspace, r, U_null, V_null, S_null
    
class SVDLinear(nn.Module):
    def __init__(self,  name, cfg, in_features, out_features, linear_layer, bias=False, mode='svft', svd_select='minor', vm_block_list=None, vm_weight=0.0):
        super().__init__()
        self.name = name
        # 먼저 기본 속성들 설정
        self.mode = mode

        self.current_abs_std = None
        
        self.svft_linear = nn.Linear(in_features, out_features, bias)
        self.svft_linear.weight.data = linear_layer.weight.data.detach().clone()
        if bias:
            self.svft_linear.bias.data = linear_layer.bias.data.detach().clone() if bias else None
        self.svft_weight = self.svft_linear.weight
        self.svft_bias = self.svft_linear.bias

        self.svft_weight.requires_grad = False
        if self.svft_bias is not None:
            self.svft_bias.requires_grad = False

        del self.svft_linear

        U, S, Vh, r, U_null, V_null, S_null = get_svd(self.svft_weight)
        self.full_rank = r
        self.svft_U = nn.Parameter(U.cpu(), requires_grad=False)
        self.svft_S = nn.Parameter(S.cpu(), requires_grad=True)
        self.svft_Vh = nn.Parameter(Vh.cpu(), requires_grad=False)
        
        self.original_S_list = nn.ParameterList()
        self.original_S_list.append(nn.Parameter(S.cpu().clone(), requires_grad=False))        

        self.full_rank = r
        
        if self.mode in ['svft']:
            pass
        elif self.mode in ['svft_variance_maximization']:
            self.vm_block_list = vm_block_list
            self.vm_weight = vm_weight

        self.dynamic_mode = 'default_adapt'
        self.temp = cfg.SVD.TEMP

    def variance_maximization(self, vm_ratio=0.05):
        if self.current_abs_std is None:
            return None
        lowk = min(int(self.current_abs_std.size(0) * vm_ratio), self.current_abs_std.size(0) -1)
        sorted_abs_std, sorted_idx = torch.sort(self.current_abs_std, dim=0, descending=False)
        filter = self.current_abs_std < sorted_abs_std[lowk]
        vm_loss = self.current_abs_std[filter].mean()
        return vm_loss
    
    def get_weight_normal(self):
        U, Vh = self.svft_U, self.svft_Vh
        S = torch.diag(self.svft_S)
        return U @ S @ Vh

    
    def get_S(self):
        if self.get_adapt_mode() == 'no_adapt':
            return self.svft_S
        elif self.get_adapt_mode() == 'adapt':
            return self.svft_S
        else:
            raise ValueError(f"Unknown forward mode: {self.dynamic_mode}")

    def forward(self, input):
        if self.get_adapt_mode() == 'no_adapt':
            weight = self.svft_weight
        elif self.get_adapt_mode() == 'adapt':
            weight = self.get_weight_normal()
        else:
            raise ValueError(f"Unknown forward mode: {self.dynamic_mode}")
        
        if self.svft_bias is not None:
            output = F.linear(input, weight, self.svft_bias)
        else:
            output =  F.linear(input, weight)

        if self.mode in ['svft_variance_maximization'] and self.vm_weight > 0.0:
            calculate_vm_loss = False
            for block_name in self.vm_block_list:
                if block_name in self.name:
                    calculate_vm_loss = True
            if calculate_vm_loss:
                norm_input = input.norm(p=2, dim=(2), keepdim=True)
                normalized_input = input / norm_input
                normalized_activated = F.linear(normalized_input, self.svft_Vh)
                abs_std = normalized_activated.std(dim=(0, 1))
                self.current_abs_std = abs_std
            else:
                self.current_abs_std = None

        return output

    def save_statistics(self):
        self.abs_std_list.append(self.abs_std)
        self.abs_mean_list.append(self.abs_mean)
        self.abs_std_divide_mean_list.append(self.abs_std_divide_mean)
    
    def update_statistics(self, abs_std, abs_mean, abs_std_divide_mean, momentum=0.9):
        if self.abs_std is None and self.abs_mean is None and self.abs_std_divide_mean is None:
            self.abs_std = abs_std
            self.abs_mean = abs_mean
            self.abs_std_divide_mean = abs_std_divide_mean
        else:
            self.abs_std = self.abs_std * momentum + abs_std * (1 - momentum)
            self.abs_mean = self.abs_mean * momentum + abs_mean * (1 - momentum)
            self.abs_std_divide_mean = self.abs_std_divide_mean * momentum + abs_std_divide_mean * (1 - momentum)

    def get_adapt_mode(self):
        if self.dynamic_mode in ['mean_no_adapt', 'continual_no_adapt', 'top1_no_adapt', 'mix_no_adapt']:
            return 'no_adapt'
        elif self.dynamic_mode in ['mean_adapt', 'continual_adapt', 'default_adapt', 'top1_adapt', 'mix_adapt']:
            return 'adapt'
        else:
            raise ValueError(f"Unknown forward mode: {self.dynamic_mode}")

    def normalized_inverse(self, weights, eps=1e-6):
        weights = torch.tensor(weights)
        inv = 1.0 / (weights + eps)
        return inv / inv.sum()
    
    def set_forward_mode(self, mode, idx=0, domain_shift_value_list=None):
        self.selected_domain_idx = idx
            
        self.dynamic_mode = mode
        target_source_list = [(self.svft_S, self.original_S_list)]

        for target_param, source_list in target_source_list:
            new_val, req_grad = self._apply_mode_generic(source_list, mode, idx, domain_shift_value_list)
            target_param.data.copy_(new_val)
            target_param.requires_grad = req_grad

    def _apply_mode_generic(
            self,
            original_list: torch.nn.ParameterList,   # [P₀, P₁, …, Pₙ]
            mode: str,
            idx: int = 0,
            domain_shift_value_list: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, bool]:
        """
        original_list : domain 별로 저장된 ParameterList
        mode          : continual_no_adapt, mix_adapt … 같은 forward mode
        idx           : top-k 선택 시 사용할 index
        return        : (새 parameter tensor, requires_grad 여부)
        """
        device = original_list[0].device
        # 1) delta = Pᵢ − P₀ 모음
        delta_stack = torch.stack(
            [p.to(device) - original_list[0].to(device) for p in original_list],
            dim=0)

        # 2) 모드별 파라미터 계산
        if mode.endswith('_no_adapt'):
            # 단 한 번도 미분 안 함
            if 'continual' in mode:   new_p = original_list[-1]
            elif 'top1'      in mode: new_p = original_list[idx]
            elif 'mean'      in mode: new_p = original_list[0] + delta_stack.mean(0)
            elif 'mix'       in mode:
                w = torch.softmax(-domain_shift_value_list / self.temp, dim=0)
                new_p = original_list[0] + (w[:, None] * delta_stack).sum(0)
            else:                     new_p = original_list[0]   # original
            return new_p, False

        elif mode.endswith('_adapt'):
            if 'continual' in mode:   new_p = original_list[-1]
            elif 'top1'      in mode: new_p = original_list[idx]
            elif 'mean'      in mode: new_p = original_list[0] + delta_stack.mean(0)
            elif 'mix'       in mode:
                w = torch.softmax(-domain_shift_value_list / self.temp, dim=0)
                new_p = original_list[0] + (w[:, None] * delta_stack).sum(0)
            else:                     new_p = original_list[0]   # original
            return new_p, True
        else:
            raise ValueError(f'Unknown mode {mode}')


    def save_singular_value(self):
        if self.get_adapt_mode() == 'no_adapt':
            pass
        else:
            self.original_S_list.append(nn.Parameter(self.svft_S.detach().clone().cpu(), requires_grad=False))

    def get_parameter_dict(self):
        """Get all trainable parameters for current mode"""
        return {
                'original_S_list': self.original_S_list.state_dict(),
                'mode': self.mode
            }
    
    def load_parameter_dict(self, param_dict):
        """Load parameters from dictionary"""
        if param_dict['mode'] != self.mode:
            raise ValueError(f"Mode mismatch: expected {self.mode}, got {param_dict['mode']}")
        self.original_S_list.load_state_dict(param_dict['original_S_list'])

def convert_linear_to_svdlinear(
    model: nn.Module,
    target_replace_module: List[str] = ["CrossAttention", "Attention"],
    mode: str = 'svft',
    svd_select: str = 'minor',
    cfg = None,
    vm_block_list: List[str] = [],
    vm_weight: float = 0.0,
):
    """
    inject vida into model, and returns vida parameter groups.
    """

    require_grad_params = []
    names = []

    for outer_name, _module in list(model.model.named_modules()):
        if _module.__class__.__name__ in target_replace_module:
            for name, _child_module in _module.named_modules():
                if _child_module.__class__.__name__ == "Linear":
                    current_name = f"{outer_name}.{name}"
                    weight = _child_module.weight
                    bias = _child_module.bias

                    _tmp = SVDLinear(
                        current_name,
                        cfg,
                        _child_module.in_features,
                        _child_module.out_features,
                        _child_module,
                        _child_module.bias is not None,
                        mode=mode,
                        svd_select=svd_select,
                        vm_block_list=vm_block_list,
                        vm_weight=vm_weight,
                    )
                    # switch the module
                    _module._modules[name] = _tmp
                    for param in _tmp.parameters():
                        param.requires_grad = False
                    if mode in ['svft']:
                        require_grad_params.append(_tmp.svft_S)
                        _tmp.svft_S.requires_grad = True
                        if cfg.SVD.BIAS:
                            require_grad_params.append(_tmp.svft_bias)
                            _tmp.svft_bias.requires_grad = True
                    elif mode in ['svft_variance_maximization']:
                        require_grad_params.append(_tmp.svft_S)
                        _tmp.svft_S.requires_grad = True
                    else:
                        raise ValueError(f"Unknown mode: {mode}")                


                    names.append(name)

    return require_grad_params, names