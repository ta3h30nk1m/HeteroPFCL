from federated_methods.task_id import LLaVATrainerTaskId
from federated_methods.fedavg import LLaVATrainerFEDAVG, get_grad_penultimate
from federated_methods.fedours import LLaVATrainerOURS
import contextlib
import copy
import functools
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import RandomSampler
from packaging import version
from torch import nn
from utils.train_utils import load_deepspeed
from transformers.utils import logging
import sys, os, time, shutil, datetime
import math
from typing import Optional, Dict, Union, Any
from transformers.integrations.tpu import tpu_spmd_dataloader
from transformers.trainer_utils import (
    HPSearchBackend,
    TrainOutput,
    has_length,
    speed_metrics,
)
from transformers.trainer_pt_utils import get_model_param_count, get_dataloader_sampler, reissue_pt_warnings
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from transformers.integrations.deepspeed import deepspeed_init, deepspeed_load_checkpoint
from transformers import Trainer
import bitsandbytes
from transformers.trainer import (
    is_sagemaker_mp_enabled, 
    _is_peft_model, 
    TRAINER_STATE_NAME,
    is_torch_xla_available,
    is_accelerate_available,
    is_deepspeed_available,
    get_parameter_names,
    ALL_LAYERNORM_LAYERS, SCHEDULER_NAME
)
from transformers.integrations import hp_params
from transformers.trainer_callback import TrainerState, ExportableState
from transformers.training_args import ParallelMode

if is_accelerate_available():
    from accelerate import skip_first_batches
    from accelerate import __version__ as accelerate_version
    from accelerate.utils import (
        DistributedType
    )

    DATA_SAMPLERS = [RandomSampler]
    if version.parse(accelerate_version) > version.parse("0.23.0"):
        from accelerate.data_loader import SeedableRandomSampler

        DATA_SAMPLERS += [SeedableRandomSampler]
    if is_deepspeed_available():
        from accelerate.utils import DeepSpeedSchedulerWrapper

import warnings

logger = logging.get_logger(__name__)

def fedours_layerwise_create_trainer(model, tokenizer, training_args, data_module, extra_state_dict_dict):
    task_id = extra_state_dict_dict['task_id'] if 'task_id' in extra_state_dict_dict else None
    ema_ratio = training_args.ema_ratio
    training_args.max_seq_length = training_args.model_max_length
    training_args.packing=False
    trainer = LLaVATrainerOURS_Layerwise(model=model,
        tokenizer=tokenizer,
        args=training_args,
        client_id = extra_state_dict_dict['client_id'],
        curr_round = extra_state_dict_dict['curr_round'],
        test_datalist=extra_state_dict_dict['test_datalist'],
        processor=extra_state_dict_dict['processor'],
        data_args=extra_state_dict_dict['data_args'],
        task_id = task_id,
        ema_ratio=ema_ratio,
        task_vector=extra_state_dict_dict['task_vector'] if 'task_vector' in extra_state_dict_dict else None,
        fisher_old=extra_state_dict_dict['fisher_old'] if 'fisher_old' in extra_state_dict_dict else None,
        fisher_freq=extra_state_dict_dict['fisher_freq'] if 'fisher_freq' in extra_state_dict_dict else 5,
        model2=extra_state_dict_dict['model2'] if 'model2' in extra_state_dict_dict else None,
        **data_module,
        )
    return trainer

def kl_loss(output, target, temp=2):
    if output.shape[-1]>3000:
        p = F.log_softmax(output / temp, dim=-1)
        q = F.softmax(target / temp, dim=-1)
    else:
        p = F.log_softmax(output / temp, dim=1)
        q = F.softmax(target / temp, dim=1)

    l_kl = F.kl_div(p, q, reduction="batchmean") #FIXME
    l_kl = l_kl * temp**2
    return l_kl

class LLaVATrainerOURS_Layerwise(LLaVATrainerOURS):
    def __init__(self, task_id, ema_ratio=0.996, task_vector=None, fisher_old=None, fisher_freq=5, model2=None,**kwargs):
        super(LLaVATrainerOURS_Layerwise, self).__init__(task_id=task_id,ema_ratio=ema_ratio,task_vector=task_vector,fisher_old=fisher_old,fisher_freq=fisher_freq,model2=model2,
                                                         **kwargs)
        
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):      
        inputs['output_hidden_states'] = True
        loss, outputs = super(LLaVATrainerOURS_Layerwise, self).compute_loss(model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch)
        # if 'feddualMultipqfullfreeze' in self.args.mode:
        layer_num = len(self.model.base_model.language_model.model.layers) // 4
        target_layers = [layer_num*1 -1,layer_num*2 -1,layer_num*3 -1,layer_num*4 -1]
        
        if 'distill' in self.args.mode:
            # self-dstill on 
            distillation_loss = 0
            for idx, target_layer in enumerate(target_layers):
                output_repr = model.module.base_model.language_model.model.layers[target_layer].mlp.down_proj.lora_C['default'](outputs.hidden_states[target_layer+1])
                start_range = 0 if idx == 0 else target_layers[idx-1]+1
                blockwise_distill_loss = 0
                for target_idx in range(start_range, target_layer):
                    target_repr = model.module.base_model.language_model.model.layers[target_idx].mlp.down_proj.lora_C['default'](outputs.hidden_states[target_idx+1].detach())
                    blockwise_distill_loss += torch.norm(output_repr[inputs['attention_mask']] - target_repr[inputs['attention_mask']], dim=-1, p=2).nanmean()
                distillation_loss += blockwise_distill_loss / (target_layer-start_range)
            loss += self.args.distill_weight * distillation_loss
            
        if 'Taskloss' in self.args.mode:
            # projection -> layernorm -> head -> loss
            labels = inputs['labels']
            blockwise_taskloss = 0
            for target_layer in target_layers:
                proj_out = model.module.base_model.language_model.model.layers[target_layer].mlp.down_proj.lora_F['default'](outputs.hidden_states[target_layer+1])
                if target_layer+1 != len(outputs.hidden_states) - 1: # last layer hidden state has already passed through norm
                    proj_out = model.module.base_model.language_model.model.norm(proj_out)
                block_logit = model.module.base_model.language_model.lm_head(proj_out)
                
                shift_logits = block_logit[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                loss_fct = nn.CrossEntropyLoss()
                blockwise_taskloss += loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1).to(shift_logits.device)
                )
            loss += self.args.taskloss_weight * blockwise_taskloss

        if 'KLloss' in self.args.mode:
            # projection -> layernorm -> head -> loss
            labels = inputs['labels']
            blockwise_taskloss = 0
            for target_layer in target_layers:
                proj_out = model.module.base_model.language_model.model.layers[target_layer].mlp.down_proj.lora_F['default'](outputs.hidden_states[target_layer+1])
                if target_layer+1 != len(outputs.hidden_states) - 1: # last layer hidden state has already passed through norm
                    proj_out = model.module.base_model.language_model.model.norm(proj_out)
                block_logit = model.module.base_model.language_model.lm_head(proj_out)
                
                shift_logits = block_logit[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()

                pred = shift_logits[shift_labels != -100]
                target = (outputs.logits[..., :-1, :].contiguous())[shift_labels != -100].detach()
                blockwise_taskloss += kl_loss(pred, target)
                
            loss += self.args.taskloss_weight * blockwise_taskloss
    
        return (loss, outputs) if return_outputs else loss
