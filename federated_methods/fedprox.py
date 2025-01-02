import torch
from federated_methods.fedavg import LLaVATrainerFEDAVG
  
def fedprox_set_state_dict(model, global_state_dict, local_state_dict_list, training_args):
    return {
        'global_state': global_state_dict,
    }

def fedprox_create_trainer(model, tokenizer, training_args, data_module, extra_state_dict_dict):
    trainer = LLaVATrainerFEDPROX(model=model,
        tokenizer=tokenizer,
        args=training_args,
        packing=True,
        max_seq_length=training_args.model_max_length,
        **data_module,
        client_id = extra_state_dict_dict['client_id'],
        curr_round = extra_state_dict_dict['curr_round'],
        global_state=extra_state_dict_dict['global_state'],
        )
    return trainer

class LLaVATrainerFEDPROX(LLaVATrainerFEDAVG):
    def __init__(self, global_state, **kwargs):
        super(LLaVATrainerFEDPROX, self).__init__(**kwargs)
        self.global_state = global_state
        self.mu = 0.01
    
    def compute_loss(self, model, inputs, return_outputs=False):

        return_values = super(LLaVATrainerFEDAVG, self).compute_loss(model, inputs, return_outputs=return_outputs)

        if return_outputs:
            loss, outputs = return_values
        else:
            loss = return_values

        # Apply FedProx Loss
        for name, param in model.module.named_parameters():
            # only trainable parameters
            if not param.requires_grad:
                continue
            else:
                loss += self.mu / 2 * torch.norm(param - self.global_state[name].cuda()) ** 2

        return (loss, outputs) if return_outputs else loss