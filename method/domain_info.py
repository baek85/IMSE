# from copy import deepcopy

import torch
import torch.nn as nn
import logging
from .decompose_svd import SVDLinear

logger = logging.getLogger(__name__)



class DomainInfo(nn.Module):
    """SAR online adapts a model by Sharpness-Aware and Reliable entropy minimization during testing.
    Once SARed, a model adapts itself by updating on every forward.
    """
    def __init__(self, cfg, model):
        super().__init__()
        self.frozen_model = model
        
        self.encountered_test_domains = 0
        self.saved_test_domains = 0

        self.cfg = cfg
        
        self.domain_detect_on = True
        self.shift_detection_threshold = self.cfg.SVD.DSS_K
            
        self.feature_layers = getattr(cfg.SVD, 'FEATURE_LAYER')
        self.domain_var = None
        self.domain_mean = None
        self.num_step = 0

        self.domain_shift_value_list = None

        self.domain_infos = {layer: [] for layer in self.feature_layers}
        self.domain_var = {}
        self.domain_mean = {}
        for layer in self.feature_layers:
            self.domain_var[layer] = None
            self.domain_mean[layer] = None

        self.ema_domain_shift_value = None
        self.relative_domain_shift_value = None
        self.update_ema = 0.2
        self.dss_ema = 0.2

        self.predicted_changes = []
        self.dss_list = []

        self.domain_optimizer_states = []
        self.original_optimizer_state = None
        self.test_batch_size = self.cfg.TEST.BATCH_SIZE

    def forward(self, x, update=True, no_dss=False):
        embed_feature_dict = self.forward_embedding_patch_embed(x)
        self.domain_shift_value_list_dict = {}
        self.domain_shift_value = 0.0
        for layer, embed_features in embed_feature_dict.items(): 
            emb_var, emb_mean = embed_features.var(dim=(0,1)), embed_features.mean(dim=(0,1))
            
            if not no_dss:
                if self.domain_var[layer] is not None and self.domain_mean[layer] is not None:
                    domain_shift_value = self._calculate_domain_shift(
                        self.domain_var[layer],
                        self.domain_mean[layer],
                        emb_var,
                        emb_mean,
                        distance=self.cfg.SVD.DOMAIN_DISTANCE
                    )
                else:
                    domain_shift_value = 0.0
                
                self.domain_shift_value = domain_shift_value
                
                if self.ema_domain_shift_value is None:
                    if domain_shift_value != 0.0:
                        self.ema_domain_shift_value = domain_shift_value
                else:
                    self.relative_domain_shift_value = domain_shift_value / self.ema_domain_shift_value
                    self.ema_domain_shift_value = 0.8 * self.ema_domain_shift_value + (1 - 0.8) * domain_shift_value

                if self.domain_detect_on:
                    if self.domain_detect_on and domain_shift_value > self.shift_detection_threshold:
                        logger.info("Domain shift detected using domain information relative domain shift value: {}".format(self.domain_shift_value))
                        self.domain_shift_detected()
                        self.predicted_changes.append(len(self.dss_list))
                        print(f"predicted changes: {self.predicted_changes}")
                self.dss_list.append(domain_shift_value)

            if embed_features.shape[0] < self.test_batch_size:
                pass
            else:
                self._update_domain_info(layer, emb_var, emb_mean)
            if self.num_step == 0 and len(self.domain_infos[layer]) >= 1:
                domain_shift_value_list = []
                for i in range(len(self.domain_infos[layer])):
                    domain_shift_value = self._calculate_domain_shift(
                        self.domain_infos[layer][i][0],
                        self.domain_infos[layer][i][1],
                        emb_var,
                        emb_mean,
                        distance=self.cfg.SVD.RETRIEVAL_DISTANCE
                    )
                    domain_shift_value_list.append(domain_shift_value)
            else:
                pass

            if self.num_step == 0 and len(self.domain_infos[layer]) > 1:
                domain_shift_value_list = torch.stack(domain_shift_value_list)
                self.domain_shift_value_list_dict[layer] = domain_shift_value_list
                
        self.num_step += 1
        return None

    @torch.no_grad()
    def forward_embedding(self, x):
        features = {}
        if self.cfg.MODEL.ARCH == 'VITB_augreg_in21k_ours':
            x = self.frozen_model.normalize(x)
            x = self.frozen_model.model.patch_embed(x)
            x = self.frozen_model.model._pos_embed(x)
            x = self.frozen_model.model.norm_pre(x)
            if -1 in self.feature_layers:
                features[-1] = x[:, 1:].clone()
            for i, block in enumerate(self.frozen_model.model.blocks):
                x = block(x)
                if i in self.feature_layers:
                    features[i] = x[:, 1:].clone()
        else:
            x = self.frozen_model.patch_embed(x)
            x = self.frozen_model._pos_embed(x)
            x = self.frozen_model.norm_pre(x)
            if -1 in self.feature_layers:
                features[-1] = x[:, 1:].clone()
            
            for i, block in enumerate(self.frozen_model.blocks):
                x = block(x)
                if i in self.feature_layers:
                    features[i] = x[:, 1:].clone()
        return features

    @torch.no_grad()
    def forward_embedding_patch_embed(self, x):
        features = {}
        if self.cfg.MODEL.ARCH == 'VITB_augreg_in21k_ours':
            x = self.frozen_model.normalize(x)
            x = self.frozen_model.model.patch_embed(x)
            x = self.frozen_model.model._pos_embed(x)
            x = self.frozen_model.model.norm_pre(x)
            features[-1] = x[:, 1:].clone()
        else:
            x = self.frozen_model.patch_embed(x)
            x = self.frozen_model._pos_embed(x)
            x = self.frozen_model.norm_pre(x)
            features[-1] = x[:, 1:].clone()
            
        return features

    @torch.no_grad()
    def _calculate_domain_shift(self, domain_var, domain_mean, cur_var, cur_mean, distance='kl_div'):
        if distance == 'kl_div':
            d1 = (domain_var + (domain_mean - cur_mean) ** 2) / 2. / cur_var - 0.5
            d2 = (cur_var + (domain_mean - cur_mean) ** 2) / 2. / domain_var - 0.5
            return torch.mean((d1+d2))
        elif distance == 'mean':
            d = (domain_mean - cur_mean) ** 2
            return torch.mean(d)
        elif distance == 'variance':
            d = (domain_var - cur_var) ** 2
            return torch.mean(d)
        elif distance == 'l2_distance':
            d = (domain_var - cur_var) ** 2 + (domain_mean - cur_mean) ** 2
            return torch.mean(d)

    @torch.no_grad()
    def _update_domain_info(self, layer, emb_var, emb_mean):
        if self.domain_var[layer] is None:
            self.domain_var[layer], self.domain_mean[layer] = emb_var, emb_mean
        else:
            self.domain_var[layer] = 0.8 * self.domain_var[layer] + 0.2 * emb_var
            self.domain_mean[layer] = 0.8 * self.domain_mean[layer] + 0.2 * emb_mean

    def reset(self):
        pass
    
    def reset_soft(self):
        pass

    def save_domain_info(self, save_singular_value=True, save_optimizer_state=True):
        """Save the model and optimizer states."""
        if save_singular_value:
            for name, module in self.frozen_model.named_modules():
                if isinstance(module, SVDLinear):
                    module.save_singular_value()
            logger.info("save singular value")

        for layer in self.feature_layers:
            self.domain_infos[layer].append((self.domain_var[layer], self.domain_mean[layer]))

        if save_optimizer_state:
            self.domain_optimizer_states.append(self.optimizer.state_dict())

        self.reset_domain_info()

    def reset_domain_info(self):
        for layer in self.feature_layers:
            self.domain_var[layer] = None
            self.domain_mean[layer] = None
        self.num_step = 0
    
    def domain_shift_detected(self):
        """Check if domain shift is detected."""
        self.num_step = 0

        self.encountered_test_domains += 1
        self.save_domain_info()
        self.saved_test_domains += 1

        logger.info(f"domain shift detected, save expert")
        logger.info(f"encountered num_experts: {self.encountered_test_domains}")
        if self.saved_test_domains >= self.cfg.SVD.NUM_SEEN_TEST_DOMAINS:
            logger.info(f"encountered num_experts: {self.encountered_test_domains}")
            logger.info(f"saved num_experts: {self.saved_test_domains}")
        
