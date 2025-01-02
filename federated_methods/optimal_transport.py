import torch
from federated_methods.fedavg import LLaVATrainerFEDAVG
import copy
from transformers.trainer import unwrap_model, _is_peft_model, MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from easydict import EasyDict as edict
from utils.optimal_transport import get_wassersteinized_layers_modularized
    
def OT_create_trainer(model, tokenizer, training_args, data_module, extra_state_dict_dict):
    trainer = LLaVATrainerOT(model=model,
        tokenizer=tokenizer,
        args=training_args,
        packing=True,
        max_seq_length=training_args.model_max_length,
        client_id=extra_state_dict_dict['client_id'],
        curr_round=extra_state_dict_dict['curr_round'],
        **data_module,
        )
    return trainer


def _compute_transport_matrix(old_prompt_weight, new_prompt_weight):
    activation_based = False
    kwargs = edict()
    
    old_prompt_reference_weight = copy.deepcopy(new_prompt_weight)
    progress = 0

    # put params into old prompt module
    for pp in list(old_prompt_reference_weight.values()):
        if not pp.requires_grad:
            continue
        cand_params = old_prompt_weight[
            progress : progress + torch.tensor(pp.size()).prod()
        ].view(pp.size())
        progress += torch.tensor(pp.size()).prod()
        pp.data = cand_params

    aligned_new_working_memory, cur_T_vars = get_wassersteinized_layers_modularized(
        models=[new_prompt_weight, old_prompt_reference_weight],
        activation_based=activation_based,
        **kwargs,
    )
    return aligned_new_working_memory, cur_T_vars

class LLaVATrainerOT(LLaVATrainerFEDAVG):
    def __init__(self, **kwargs):
        super(LLaVATrainerOT, self).__init__(**kwargs)
        
        self.old_weights = {k: t.detach().cpu().clone() for k, t in self.model.named_parameters() if t.requires_grad}
    
    def _inner_training_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        output = super()._inner_training_loop(batch_size=batch_size, args=args,resume_from_checkpoint=resume_from_checkpoint,ignore_keys_for_eval=ignore_keys_for_eval)
        
        # Optimal Transport Combination within client
        if self.curr_round > 0:
            cur_weights = {k: t.detach().cpu().clone() for k, t in self.model.named_parameters() if t.requires_grad}
            
            ot_mat, cur_T_var = _compute_transport_matrix(self.old_weights, cur_weights)
            
            # aggregate
            # if self._use_ot and self.attribution_aware_fusion:
            #     self._align_attribution()

            # if self.attribution_aware_fusion:
            #     global_importance = self._get_global_attribution(normalize=True).type_as(
            #         self.prev_reference_prompt_memory
            #     )
            #     current_importance = self._get_current_attribution(normalize=True).type_as(
            #         self.prev_reference_prompt_memory
            #     )
            # fusion weight
            alpha = 0.5
            for key in cur_weights.keys():
                weight = torch.ones_like(cur_weights[key]) * alpha
                # if self.attribution_aware_fusion:
                #     weight = (torch.ones_like(self.prev_reference_prompt_memory) * alpha) + (
                #         (current_importance - global_importance) * alpha
                #     )
                # cur_weights[key] = cur_weights[key] + weight * (self.old_weights[key] - cur_weights[key])
                cur_weights[key] = cur_weights[key] + weight * (ot_mat[key] - cur_weights[key])
            self.model.load_state_dict(cur_weights, strict=False)
            
        return output
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        
        if 'prompt' in inputs:
            text_prompt = inputs.pop('prompt')
        else:
            text_prompt = None
        outputs = model(**inputs, prompt=text_prompt) if text_prompt else model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            unwrapped_model = unwrap_model(model)
            if _is_peft_model(unwrapped_model):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss