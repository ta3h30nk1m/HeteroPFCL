import torch
from torch import nn
from federated_methods.fedavg import LLaVATrainerFEDAVG
            
def fedsim_create_trainer(model, tokenizer, training_args, data_module, extra_state_dict_dict):
    training_args.max_seq_length = training_args.model_max_length
    training_args.packing=False
    trainer = LLaVATrainerFEDSIM(model=model,
        tokenizer=tokenizer,
        args=training_args,
        client_id = extra_state_dict_dict['client_id'],
        curr_round = extra_state_dict_dict['curr_round'],
        test_datalist=extra_state_dict_dict['test_datalist'],
        processor=extra_state_dict_dict['processor'],
        data_args=extra_state_dict_dict['data_args'],
        task_vector=extra_state_dict_dict['task_vector'] if 'task_vector' in extra_state_dict_dict else None,
        fisher_old=extra_state_dict_dict['fisher_old'] if 'fisher_old' in extra_state_dict_dict else None,
        fisher_freq=extra_state_dict_dict['fisher_freq'] if 'fisher_freq' in extra_state_dict_dict else 5,
        model2=extra_state_dict_dict['model2'] if 'model2' in extra_state_dict_dict else None,
        **data_module,
        )
    return trainer


def fedsim_blockwise_aggregate_state_dict(global_state_dict_list, local_state_dict_list, selected_ids, num_selection, training_args, **kwargs):
    # only aggregate the local models with the same architecture
    model_ids = kwargs['model_ids']
    layer_index = kwargs['LAYER_INDEX']
    
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


class LLaVATrainerFEDSIM(LLaVATrainerFEDAVG):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        
        # global forward
        model.module.set_state('lora1')
        _, outputs = super(LLaVATrainerFEDSIM, self).compute_loss(model, inputs, return_outputs=True,num_items_in_batch=num_items_in_batch)     
        # local forward
        model.module.set_state('lora2')
        _, local_outputs = super(LLaVATrainerFEDSIM, self).compute_loss(model, inputs, return_outputs=True,num_items_in_batch=num_items_in_batch) 
        
        final_logits = outputs['logits'] + local_outputs['logits']
        labels = inputs['labels']
        # Shift so that tokens < n predict n
        shift_logits = final_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = nn.CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels) 
        
        return (loss, outputs) if return_outputs else loss