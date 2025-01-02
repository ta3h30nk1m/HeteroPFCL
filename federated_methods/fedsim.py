import torch
from torch import nn
# from models.llava.llava_trainer import LLaVATrainer
from federated_methods.fedavg import LLaVATrainerFEDAVG
from federated_methods.task_id import LLaVATrainerTaskId

import copy
from models.duallora.dualloralayer import DualLoraLayer
from models.dual_ia3.dual_ia3_layer import DualIA3Layer
def fedsim_set_state_dict(model, global_state_dict, local_state_dict_list, training_args):
    # layers_to_del = layer_num[-len(layer_num)//2:]
    keys_to_del = []
    for k in global_state_dict.keys():
        if 'lora2' in k or 'local_mm_projector' in k or 'ia3_l_2' in k:
            keys_to_del.append(k)
    for k in keys_to_del:
        del global_state_dict[k]
    
    # local_keys_to_del = []
    # for k in local_state_dict_list[0].keys():
    #     if 'lora1' in k or 'global_mm_projector' in k:
    #         local_keys_to_del.append(k)
    # for client_id in range(training_args.num_clients):
    #     for k in local_keys_to_del:
    #         del local_state_dict_list[client_id][k]
    
    return {'global_state':global_state_dict}
            
def fedsim_create_trainer(model, tokenizer, training_args, data_module, extra_state_dict_dict):
    trainer = LLaVATrainerFEDSIM(model=model,
        tokenizer=tokenizer,
        args=training_args,
        packing=True,
        max_seq_length=training_args.model_max_length,
        client_id = extra_state_dict_dict['client_id'],
        curr_round = extra_state_dict_dict['curr_round'],
        task_id = extra_state_dict_dict['task_id'] if 'task_id' in extra_state_dict_dict else None,
        **data_module,
        )
    return trainer


class LLaVATrainerFEDSIM(LLaVATrainerTaskId):
    def compute_loss(self, model, inputs, return_outputs=False):
        
        # global forward
        # for name, module in model.module.named_modules():
            # if isinstance(module, DualLoraLayer) or isinstance(module,DualIA3Layer):
        model.module.set_state('lora1')
        _, outputs = super(LLaVATrainerFEDSIM, self).compute_loss(model, copy.deepcopy(inputs), return_outputs=True)     
        # local forward
        # for name, module in model.module.named_modules():
        #     if isinstance(module, DualLoraLayer) or isinstance(module,DualIA3Layer):
        model.module.set_state('lora2')
        _, local_outputs = super(LLaVATrainerFEDSIM, self).compute_loss(model, inputs, return_outputs=True) 
        
        final_logits = outputs['logits'] + local_outputs['logits']
        labels = model.module.labels
        # Shift so that tokens < n predict n
        shift_logits = final_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = nn.CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, model.module.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels) 
        
        return (loss, outputs) if return_outputs else loss

    def _inner_training_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        self.model.activate_all()
        
        output = super()._inner_training_loop(batch_size=batch_size, args=args,resume_from_checkpoint=resume_from_checkpoint,trial=trial,ignore_keys_for_eval=ignore_keys_for_eval)
        
        return output