from torch import nn
from federated_methods.fedavg import LLaVATrainerFEDAVG
from transformers.utils import logging
from transformers import Trainer
from transformers.trainer import (
    is_sagemaker_mp_enabled, 
    _is_peft_model,
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES, unwrap_model
)

logger = logging.get_logger(__name__)
    
def task_id_create_trainer(model, tokenizer, training_args, data_module, extra_state_dict_dict):
    task_id = extra_state_dict_dict['task_id'] if 'task_id' in extra_state_dict_dict else None
    trainer = LLaVATrainerTaskId(model=model,
        tokenizer=tokenizer,
        args=training_args,
        packing=True,
        max_seq_length=training_args.model_max_length,
        **data_module,
        task_id=task_id,
        client_id = extra_state_dict_dict['client_id'],
        curr_round = extra_state_dict_dict['curr_round'],
        )
    return trainer


class LLaVATrainerTaskId(LLaVATrainerFEDAVG):
    def __init__(self, task_id=None, **kwargs):
        super(LLaVATrainerTaskId, self).__init__(**kwargs)
        
        self.task_id = task_id
    
    def compute_loss(self, model, inputs, return_outputs=False):
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        ##############################################################################################################
        if 'prompt' in inputs:
            text_prompt = inputs.pop('prompt')
        else:
            text_prompt = None
        
        if self.task_id is not None:
            outputs = model(**inputs, task_id=self.task_id, prompt=text_prompt) if text_prompt is not None else model(**inputs, task_id=self.task_id)
        else:
            outputs = model(**inputs, prompt=text_prompt) if text_prompt is not None else model(**inputs)
        ##############################################################################################################
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
    
    def _inner_training_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        output = super()._inner_training_loop(batch_size=batch_size, args=args,resume_from_checkpoint=resume_from_checkpoint,trial=trial,ignore_keys_for_eval=ignore_keys_for_eval)
        
        # remove momentum for l2p before saving
        if 'L2P' in self.args.mode and self.task_id is not None:
            for key in self.optimizer.state.keys():
                if 'exp_avg' not in self.optimizer.state[key]:
                    continue
                self.optimizer.state[key]['exp_avg'][self.optimizer.state[key]['exp_avg']!=0] = 0.0
                self.optimizer.state[key]['exp_avg_sq'][self.optimizer.state[key]['exp_avg_sq']!=0] = 0.0

        if self.args.save_optim:
            output_dir = f'client_states_{self.args.note}/client_{self.client_id}/'
            self._save_optimizer_and_scheduler(output_dir)

        return output
    
    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            # decay_parameters = [name for name in decay_parameters if "bias" not in name]
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
            ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

        return self.optimizer