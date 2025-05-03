from federated_methods.task_id import LLaVATrainerTaskId
from federated_methods.fedavg import LLaVATrainerFEDAVG, get_grad_penultimate
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

from utils.train_utils import make_supervised_data_module, get_peft_state_maybe_zero_3, get_peft_state_non_lora_maybe_zero_3
import json
import random
import gc

def TAKFL_aggregate_state_dict(global_state_dict_list, local_state_dict_list, selected_ids, num_selection, training_args, **kwargs):
    model_ids = kwargs['model_ids']
    models = kwargs['models']
    layer_index = kwargs['LAYER_INDEX']
    
    # fedavg first
    if 'Multi' in training_args.mode:
        for model_id, homo_client_ids in model_ids.items():
            global_state_dict = global_state_dict_list[homo_client_ids[0]]
            
            # only use active clients
            active_homo_ids = [id for id in homo_client_ids if id in selected_ids]
            
            cur_layer_num = []
            for k in global_state_dict.keys():
                if 'layers.' in k:
                    cur_layer_num.append(int(k.split('.')[layer_index]))
            cur_layer_num = sorted(list(set(cur_layer_num)))
            if 'Multi05' in training_args.mode:
                cur_layer_num = [len(cur_layer_num)//2 -1, len(cur_layer_num) -1]
            elif 'Multi' in training_args.mode:
                cur_layer_num = [len(cur_layer_num)//4 -1,len(cur_layer_num)//2 -1, (len(cur_layer_num)//4) * 3 -1,len(cur_layer_num) -1]
            else:
                raise ValueError('wrong mode')
            
            for name in global_state_dict.keys():
                new_param = 0
                target_key = name
                splited = target_key.split('.')
                if int(splited[layer_index]) in cur_layer_num:
                    if 'lora_P' not in target_key and 'lora_Q' not in target_key:
                        continue
                    for id in range(training_args.num_clients):
                        splited = target_key.split('.')
                        # if layer number is different
                        layer_num = []
                        for k in local_state_dict_list[id].keys():
                            if 'layers.' in k:
                                layer_num.append(int(k.split('.')[layer_index]))
                        
                        if 'Multi05' in training_args.mode:
                            layer_num = len(set(layer_num)) // 2
                            target_layers = [layer_num*1 -1,layer_num*2 -1]
                        elif 'Multi' in training_args.mode:
                            layer_num = len(set(layer_num)) // 4
                            target_layers = [layer_num*1 -1,layer_num*2 -1,layer_num*3 -1,layer_num*4 -1]
                        if cur_layer_num[-1] != target_layers[-1]: # if different size
                            idx = cur_layer_num.index(int(splited[layer_index]))
                            splited[layer_index] = str(target_layers[idx])
                            new_target_key = '.'.join(splited)
                        else:
                            new_target_key = target_key
                    
                        new_param += local_state_dict_list[id][new_target_key] / len(selected_ids)
                else:
                    for id in active_homo_ids:
                        new_param += local_state_dict_list[id][target_key] / len(active_homo_ids)
                global_state_dict[name] = new_param
            for i in homo_client_ids:
                global_state_dict_list[i] = global_state_dict
    else:
        for model_id, homo_client_ids in model_ids.items():
            global_state_dict = global_state_dict_list[homo_client_ids[0]]
            
            # only use active clients
            active_homo_ids = [id for id in homo_client_ids if id in selected_ids]
            
            for key in global_state_dict.keys():
                global_state_dict[key] = sum([local_state_dict_list[client][key] / len(active_homo_ids) for client in active_homo_ids])
            for i in homo_client_ids:
                global_state_dict_list[i] = global_state_dict
    
    # get public dataset
    # data_path = "dataset/llava_finetune/llava_v1_5_mix665k_mixed.json"
    # data_path = 'chatbotIT.json'
    layer_index = kwargs['LAYER_INDEX']
    if layer_index == 5: # multimodal
        data_path = "dataset/llava_finetune/llava_v1_5_mix665k_mixed.json"
    else:
        data_path = "dataset/FS_LLM_Instruct/dolly_meta.json"
    # data_path = 'chatbotIT.json'
    public_datalist = json.load(open(data_path, "r"))[:training_args.num_serverdistill]
    random.shuffle(public_datalist)

    for model_id, homo_client_ids in model_ids.items():
        print(f"Aggregate model {model_id}")
        model = models[model_id]
        global_state_dict = global_state_dict_list[homo_client_ids[0]]
        
        # FIXME: Handle case with dual module
        with torch.no_grad():
            if 'zero3' in training_args.deepspeed:
                load_deepspeed(global_state_dict, model, strict=False)
            else:
                model.load_state_dict(global_state_dict, strict=False)  
        
        # distill from each model id
        task_vectors = []
        for teacher_model_id, teacher_client_ids in model_ids.items():
            teacher_client_ids = [id for id in teacher_client_ids if id in selected_ids]
            data_module = make_supervised_data_module(client_data=public_datalist, # sub_dataset
                                                    tokenizer=kwargs['tokenizer'],
                                                    processor=kwargs['processor'],
                                                    data_args=copy.deepcopy(kwargs['data_args']))
            trainer = takfl_create_trainer(model, kwargs['tokenizer'], training_args, data_module, models, local_state_dict_list, model_id, teacher_model_id, teacher_client_ids, kwargs)
            results = trainer.train()
        
            state_dict = get_peft_state_maybe_zero_3(
                model.named_parameters(), training_args.lora_bias
            )
            non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
                model.named_parameters()
            )
            state_dict.update(non_lora_state_dict)
            assert len(state_dict) == len(global_state_dict)
            for k,v in global_state_dict.items():
                state_dict[k] = v - global_state_dict[k]
            task_vectors.append(state_dict)
        
        for k,v in global_state_dict.items():
            global_state_dict[k] = v + sum([task_vectors[i][k] / len(task_vectors) for i in range(len(task_vectors))])
        for i in homo_client_ids:
            global_state_dict_list[i] = global_state_dict

        trainer.deepspeed.empty_partition_cache()
        trainer.accelerator.free_memory()
        del trainer
        model = model.cpu()
        gc.collect()
        torch.cuda.empty_cache()

def takfl_create_trainer(model, tokenizer, training_args, data_module, models, local_state_dict_list, cur_model_id, teacher_model_id, teacher_client_ids, extra_state_dict_dict):
    task_id = extra_state_dict_dict['task_id'] if 'task_id' in extra_state_dict_dict else None
    ema_ratio = training_args.ema_ratio
    training_args.max_seq_length = training_args.model_max_length
    training_args.packing=False
    trainer = LLaVATrainerTAKFL(model=model,
        tokenizer=tokenizer,
        args=training_args,
        client_id = extra_state_dict_dict['client_id'],
        curr_round = extra_state_dict_dict['curr_round'],
        test_datalist=extra_state_dict_dict['test_datalist'],
        processor=extra_state_dict_dict['processor'],
        data_args=extra_state_dict_dict['data_args'],
        
        models=models,
        local_state_dict_list=local_state_dict_list,
        cur_model_id=cur_model_id,
        teacher_model_id=teacher_model_id,
        teacher_client_ids=teacher_client_ids,
        
        **data_module,
        )
    return trainer

def kl_loss(output, target, temp=2):
    # if output.shape[-1]>3000:
    #     p = F.log_softmax(output / temp, dim=-1)
    #     q = F.softmax(target / temp, dim=-1)
    # else:
    p = F.log_softmax(output / temp, dim=1)
    q = F.softmax(target / temp, dim=1)

    l_kl = F.kl_div(p, q, reduction="batchmean") #FIXME
    l_kl = l_kl * temp**2
    return l_kl

class LLaVATrainerTAKFL(LLaVATrainerFEDAVG):
    def __init__(self, models,local_state_dict_list,cur_model_id,teacher_model_id, teacher_client_ids,**kwargs):
        super(LLaVATrainerTAKFL, self).__init__(**kwargs)
        
        self.models = models
        self.local_state_dict_list = local_state_dict_list
        self.cur_model_id=cur_model_id
        self.teacher_model_id=teacher_model_id
        self.teacher_client_ids=teacher_client_ids
        self.fedavg_state_dict = get_peft_state_maybe_zero_3(self.model.named_parameters(), self.args.lora_bias)
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        target_logits = 0
        with torch.no_grad():
            cur_state_dict = get_peft_state_maybe_zero_3(model.module.named_parameters(), self.args.lora_bias)
            
            # teacher logit
            for client_id in self.teacher_client_ids:
                local_state_dict = self.local_state_dict_list[client_id]
                if self.cur_model_id == self.teacher_model_id:
                    model.module.load_state_dict(local_state_dict, strict=False)
                            
                    _, outputs = super(LLaVATrainerTAKFL, self).compute_loss(model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch)
                    target_logits += outputs.logits / len(self.teacher_client_ids)
                else:
                    model2 = self.models[self.teacher_model_id].cuda()
                    model2.load_state_dict(local_state_dict, strict=False)
                    _, outputs = super(LLaVATrainerTAKFL, self).compute_loss(model2, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch)
                    target_logits += outputs.logits / len(self.teacher_client_ids)
            if self.cur_model_id != self.teacher_model_id:
                self.models[self.teacher_model_id] = model2.cpu()

            # self logit
            model.module.load_state_dict(self.fedavg_state_dict, strict=False)
            _, outputs = super(LLaVATrainerTAKFL, self).compute_loss(model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch)
            self_logits = outputs.logits

            model.module.load_state_dict(cur_state_dict, strict=False)
            if hasattr(model.module, 'set_state'):
                model.module.set_state('lora1')
        
        _, outputs = super(LLaVATrainerTAKFL, self).compute_loss(model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch)
        
        pred_logits = outputs.logits
        
        labels = inputs['labels']
        
        shift_logits = pred_logits[..., :-1, :].contiguous()
        target_logits = target_logits[..., :-1, :].contiguous()
        self_logits = self_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        pred = shift_logits[shift_labels != -100]
        target = target_logits[shift_labels != -100].detach()
        self_target = self_logits[shift_labels != -100].detach()
        
        # no specific coefficient value in paper
        # 0.1, 0.01, 0.001
        loss = kl_loss(pred, target, temp=3) + 0.01*kl_loss(pred, self_target, temp=20)
        
        return (loss, outputs) if return_outputs else loss
    
    def _inner_training_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        self.accelerator.free_memory()
        self._train_batch_size = batch_size
        if self.args.auto_find_batch_size:
            if self.state.train_batch_size != self._train_batch_size:
                from accelerate.utils import release_memory

                (self.model_wrapped,) = release_memory(self.model_wrapped)
                self.model_wrapped = self.model

                # Check for DeepSpeed *after* the intial pass and modify the config
                if self.is_deepspeed_enabled:
                    # Temporarily unset `self.args.train_batch_size`
                    original_bs = self.args.per_device_train_batch_size
                    self.args.per_device_train_batch_size = self._train_batch_size // max(1, self.args.n_gpu)
                    self.propagate_args_to_deepspeed(True)
                    self.args.per_device_train_batch_size = original_bs
            self.state.train_batch_size = self._train_batch_size
        logger.debug(f"Currently training with a batch size of: {self._train_batch_size}")
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()
        if self.is_fsdp_xla_v2_enabled:
            train_dataloader = tpu_spmd_dataloader(train_dataloader)

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = self._train_batch_size * args.gradient_accumulation_steps * args.world_size

        len_dataloader = None
        num_train_tokens = None
        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = self.num_examples(train_dataloader)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
                # the best we can do.
                num_train_samples = args.max_steps * total_train_batch_size
                if args.include_tokens_per_second:
                    num_train_tokens = (
                        self.num_tokens(train_dataloader, args.max_steps) * args.gradient_accumulation_steps
                    )
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs
                if args.include_tokens_per_second:
                    num_train_tokens = self.num_tokens(train_dataloader) * args.num_train_epochs
        elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
            if args.include_tokens_per_second:
                num_train_tokens = self.num_tokens(train_dataloader, args.max_steps) * args.gradient_accumulation_steps
        else:
            raise ValueError(
                "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                f" {args.max_steps}"
            )

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                # nn.DataParallel(model) replicates the model, creating new variables and module
                # references registered here no longer work on other gpus, breaking the module
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
                    " (torchrun or torch.distributed.launch (deprecated))."
                )
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        # ##############################################################################################################
        # OURS:
        if hasattr(self.model, 'set_state'):
            self.model.set_state('lora1')
            self.model.activate_lora1()
        # ##############################################################################################################
        
        delay_optimizer_creation = is_sagemaker_mp_enabled() or self.is_fsdp_xla_enabled or self.is_fsdp_enabled

        # We need to reset the scheduler, as its parameters may be different on subsequent calls
        if self._created_lr_scheduler:
            self.lr_scheduler = None
            self._created_lr_scheduler = False

        if self.is_deepspeed_enabled:
            self.optimizer, self.lr_scheduler = deepspeed_init(self, num_training_steps=max_steps)

        if not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # self.state = TrainerState()
        self.state = TrainerState(
            stateful_callbacks=[
                cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
            ]
        )
        self.state.is_hyper_param_search = trial is not None
        self.state.train_batch_size = self._train_batch_size

        # Compute absolute values for logging, eval, and save if given as ratio
        if args.logging_steps is not None:
            if args.logging_steps < 1:
                self.state.logging_steps = math.ceil(max_steps * args.logging_steps)
            else:
                self.state.logging_steps = args.logging_steps
        if args.eval_steps is not None:
            if args.eval_steps < 1:
                self.state.eval_steps = math.ceil(max_steps * args.eval_steps)
            else:
                self.state.eval_steps = args.eval_steps
        if args.save_steps is not None:
            if args.save_steps < 1:
                self.state.save_steps = math.ceil(max_steps * args.save_steps)
            else:
                self.state.save_steps = args.save_steps

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            if args.gradient_checkpointing_kwargs is None:
                gradient_checkpointing_kwargs = {}
            else:
                gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs

            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

        model = self._wrap_model(self.model_wrapped)

        # as the model is wrapped, don't use `accelerator.prepare`
        # this is for unhandled cases such as
        # FSDP-XLA, SageMaker MP/DP, DataParallel, IPEX
        use_accelerator_prepare = True if model is self.model else False
        
        if use_accelerator_prepare and self.is_fsdp_enabled:
            # In case of auto_find_batch_size=True
            # Remove FSDP wrapping from sub-models.
            self.model = unwrap_model(self.model, recursive=True)

        if delay_optimizer_creation:
            if use_accelerator_prepare:
                # configure fsdp plugin for qlora if any
                self._fsdp_qlora_plugin_updates()
                if self.accelerator.mixed_precision != "fp8":
                    self.model = self.accelerator.prepare(self.model)
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # prepare using `accelerator` prepare
        if use_accelerator_prepare:
            self.model.train()
            if hasattr(self.lr_scheduler, "step"):
                if self.use_apex:
                    model = self.accelerator.prepare(self.model)
                else:
                    model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
            else:
                # to handle cases wherein we pass "DummyScheduler" such as when it is specified in DeepSpeed config.
                model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
                    self.model, self.optimizer, self.lr_scheduler
                )

        if self.is_fsdp_enabled:
            self.model = self.model_wrapped = model

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        # backward compatibility
        if self.is_deepspeed_enabled:
            self.deepspeed = self.model_wrapped

        # ckpt loading
        if resume_from_checkpoint is not None:
            if self.is_deepspeed_enabled:
                deepspeed_load_checkpoint(
                    self.model_wrapped, resume_from_checkpoint, load_module_strict=not _is_peft_model(self.model)
                )
            elif is_sagemaker_mp_enabled() or self.is_fsdp_enabled:
                self._load_from_checkpoint(resume_from_checkpoint, self.model_wrapped)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)
        #############################################################################################################

        # if self.args.save_optim and self.curr_round > 0:
        #     output_dir = f'client_states_{self.args.note}/client_{self.client_id}/'
        #     self._load_optimizer_and_scheduler(output_dir)
            
        ##############################################################################################################
        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model),
        # FSDP(Transformers Model), Dynamo Optimized Module(Transformers Model) etc.

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples:,}")
        logger.info(f"  Num Epochs = {num_train_epochs:,}")
        logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}")
        if self.args.per_device_train_batch_size != self._train_batch_size:
            logger.info(f"  Training with DataParallel so batch size has been adjusted to: {self._train_batch_size:,}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps:,}")
        logger.info(f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            self.compare_trainer_and_checkpoint_args(self.args, self.state)
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first"
                    f" {steps_trained_in_current_epoch} batches in the first epoch."
                )

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        if self.hp_name is not None and self._trial is not None:
            # use self._trial because the SigOpt/Optuna hpo only call `_hp_search_setup(trial)` instead of passing trial
            # parameter to Train when using DDP.
            self.state.trial_name = self.hp_name(self._trial)
        if trial is not None:
            assignments = trial.assignments if self.hp_search_backend == HPSearchBackend.SIGOPT else trial
            self.state.trial_params = hp_params(assignments)
        else:
            self.state.trial_params = None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()
        grad_norm: Optional[float] = None
        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        if args.eval_on_start:
            self._evaluate(trial, ignore_keys_for_eval, skip_scheduler=True)

        for epoch in range(epochs_trained, num_train_epochs):
            epoch_dataloader = train_dataloader
            if hasattr(epoch_dataloader, "set_epoch"):
                epoch_dataloader.set_epoch(epoch)
            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_dataloader)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
                self._load_rng_state(resume_from_checkpoint)

            rng_to_sync = False
            steps_skipped = 0
            if steps_trained_in_current_epoch > 0:
                epoch_dataloader = skip_first_batches(epoch_dataloader, steps_trained_in_current_epoch)
                steps_skipped = steps_trained_in_current_epoch
                steps_trained_in_current_epoch = 0
                rng_to_sync = True

            step = -1
            epoch_iterator = iter(epoch_dataloader)
            # We chunkify the epoch iterator into gradient accumulation steps `n` batches
            remainder = num_examples % args.gradient_accumulation_steps
            if remainder == 0:
                remainder = args.gradient_accumulation_steps
            update_step = -1
            total_updates = steps_in_epoch // args.gradient_accumulation_steps + 1
            for _ in range(total_updates):
                update_step += 1
                num_batches = args.gradient_accumulation_steps if update_step != (total_updates - 1) else remainder
                batch_samples, num_items_in_batch = self.get_batch_samples(epoch_iterator, num_batches)
                for i, inputs in enumerate(batch_samples):
                    step += 1
                    do_sync_step = (step + 1) % args.gradient_accumulation_steps == 0 or (step + 1) == steps_in_epoch
                    # Since we perform prefetching, we need to manually set sync_gradients
                    if not do_sync_step:
                        self.accelerator.gradient_state._set_sync_gradients(False)
                    else:
                        self.accelerator.gradient_state._set_sync_gradients(True)

                    if self.args.include_num_input_tokens_seen:
                        main_input_name = getattr(self.model, "main_input_name", "input_ids")
                        if main_input_name not in inputs:
                            logger.warning(
                                "Tried to track the number of tokens seen, however the current model is "
                                "not configured properly to know what item is the input. To fix this, add "
                                "a `main_input_name` attribute to the model class you are using."
                            )
                        else:
                            input_tokens = inputs[main_input_name].numel()
                            input_tokens = torch.tensor(input_tokens, device=self.args.device, dtype=torch.int64)
                            self.state.num_input_tokens_seen += (
                                self.accelerator.gather(input_tokens).sum().cpu().item()
                            )
                    if rng_to_sync:
                        self._load_rng_state(resume_from_checkpoint)
                        rng_to_sync = False

                    # Skip past any already trained steps if resuming training
                    if steps_trained_in_current_epoch > 0:
                        steps_trained_in_current_epoch -= 1
                        if steps_trained_progress_bar is not None:
                            steps_trained_progress_bar.update(1)
                        if steps_trained_in_current_epoch == 0:
                            self._load_rng_state(resume_from_checkpoint)
                        continue
                    elif steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.close()
                        steps_trained_progress_bar = None

                    if step % args.gradient_accumulation_steps == 0:
                        self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                    # We explicitly want to avoid relying on `accelerator.accumulate` for generation training
                    context = (
                        functools.partial(self.accelerator.no_sync, model=model)
                        if i != len(batch_samples) - 1
                        and self.accelerator.distributed_type != DistributedType.DEEPSPEED
                        else contextlib.nullcontext
                    )
                    
                    with context():
                        tr_loss_step = self.training_step(model, inputs, num_items_in_batch)

                    if (
                        args.logging_nan_inf_filter
                        and not is_torch_xla_available()
                        and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                    ):
                        # if loss is nan or inf simply add the average of previous logged losses
                        tr_loss = tr_loss + tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                    else:
                        if tr_loss.device != tr_loss_step.device:
                            raise ValueError(
                                f"Calculated loss must be on the original device: {tr_loss.device} but device in use is {tr_loss_step.device}"
                            )
                        tr_loss = tr_loss + tr_loss_step

                    self.current_flos += float(self.floating_point_ops(inputs))

                    if do_sync_step:
                        # Since we perform prefetching, we need to manually set sync_gradients to True
                        self.accelerator.gradient_state._set_sync_gradients(True)

                        # Gradient clipping
                        if args.max_grad_norm is not None and args.max_grad_norm > 0:
                            # deepspeed does its own clipping

                            if is_sagemaker_mp_enabled() and args.fp16:
                                _grad_norm = self.optimizer.clip_master_grads(args.max_grad_norm)
                            elif self.use_apex:
                                # Revert to normal clipping otherwise, handling Apex or full precision
                                _grad_norm = nn.utils.clip_grad_norm_(
                                    amp.master_params(self.optimizer),
                                    args.max_grad_norm,
                                )
                            else:
                                _grad_norm = self.accelerator.clip_grad_norm_(
                                    model.parameters(),
                                    args.max_grad_norm,
                                )

                            if (
                                is_accelerate_available()
                                and self.accelerator.distributed_type == DistributedType.DEEPSPEED
                            ):
                                grad_norm = model.get_global_grad_norm()
                                # In some cases the grad norm may not return a float
                                if hasattr(grad_norm, "item"):
                                    grad_norm = grad_norm.item()
                            else:
                                grad_norm = _grad_norm

                        self.control = self.callback_handler.on_pre_optimizer_step(args, self.state, self.control)

                        self.optimizer.step()

                        self.control = self.callback_handler.on_optimizer_step(args, self.state, self.control)
                        
                        if self.args.use_hypergradient:
                            # update self.lr_scheduler.base_lrs
                            for param_id, param_group in enumerate(self.optimizer.param_groups):
                                self.lr_scheduler.base_lrs[param_id] = param_group['lr']

                        optimizer_was_run = not self.accelerator.optimizer_step_was_skipped
                        if optimizer_was_run:
                            # Delay optimizer scheduling until metrics are generated
                            if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                                self.lr_scheduler.step()

                        model.zero_grad()
                        ##############################################################################################################
                        self.state.global_step += 1
                        self.state.epoch = epoch + (step + 1 + steps_skipped) / steps_in_epoch
                        self.control = self.callback_handler.on_step_end(args, self.state, self.control)
                        
                        ##############################################################################################################
                        
                        self._maybe_log_save_evaluate(
                            tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time
                        )
                    else:
                        self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                    # PyTorch/XLA relies on the data loader to insert the mark_step for
                    # each step. Since we are breaking the loop early, we need to manually
                    # insert the mark_step here.
                    if self.control.should_epoch_stop or self.control.should_training_stop:
                        if is_torch_xla_available():
                            xm.mark_step()
                        break
                # We also need to break out of the nested loop
                if self.control.should_epoch_stop or self.control.should_training_stop:
                    if is_torch_xla_available():
                        xm.mark_step()
                    break
            if step < 0:
                logger.warning(
                    "There seems to be not a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time)

            if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                if is_torch_xla_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sure the model has been saved by process 0.
            if is_torch_xla_available():
                xm.rendezvous("load_best_model_at_end")
            elif args.parallel_mode == ParallelMode.DISTRIBUTED:
                dist.barrier()
            elif is_sagemaker_mp_enabled():
                smp.barrier()

            self._load_best_model()

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        effective_global_step = max(self.state.global_step, 0.001)  # Avoid ZeroDivisionError
        train_loss = self._total_loss_scalar / effective_global_step

        metrics = speed_metrics(
            "train",
            start_time,
            num_samples=num_train_samples,
            num_steps=self.state.max_steps,
            num_tokens=num_train_tokens,
        )
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        run_dir = self._get_output_dir(trial)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)

        # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint and process allowed to save.
        if self.args.should_save and self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
            for checkpoint in checkpoints_sorted:
                if not os.path.samefile(checkpoint, self.state.best_model_checkpoint):
                    logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
                    shutil.rmtree(checkpoint)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        # Wait for the checkpoint to be uploaded.
        self._finish_current_push()

        # After training we make sure to retrieve back the original forward pass method
        # for the embedding layer by removing the forward post hook.
        if self.neftune_noise_alpha is not None:
            self._deactivate_neftune(self.model)


        # if self.args.save_optim:
        #     output_dir = f'client_states_{self.args.note}/client_{self.client_id}/'
        #     self._save_optimizer_and_scheduler(output_dir)
        if hasattr(self.model, 'set_state'):
            self.model.activate_all()

        return TrainOutput(self.state.global_step, train_loss, metrics)
