import nevergrad as ng
from utils.train_utils import make_supervised_data_module, load_deepspeed
import random
import json

from functools import partial
import copy
import torch
import os
from torch.utils.data import DataLoader


def get_final_weights(weights, cache):
    final_state_dict = {}
    keys = cache[0].keys()
    for i in range(len(cache)):
        lora_state_dict = cache[i]
        if i == 0:
            for key in keys:
                final_state_dict[key] = weights[i] * lora_state_dict[key]
        else:
            for key in keys:
                final_state_dict[key] = (
                    final_state_dict[key] + weights[i] * lora_state_dict[key]
                )
    return final_state_dict

def get_loss(data_module, model, batch_size):
    """
    Get the loss of the model on the example dataset. Usually the example dataset only contains a few examples.
    """
    total_num = len(data_module['train_dataset'])
    data_batch_size = total_num if batch_size is None else min(total_num, batch_size)
    
    train_dataloader = DataLoader(
        data_module['train_dataset'],
        collate_fn=data_module['data_collator'],
        batch_size=data_batch_size,
        pin_memory=True,
    )
    train_loss = 0
    with torch.no_grad():
        device = "cuda" if torch.cuda.is_available() else "cpu"
        for _, batch in enumerate(train_dataloader):
            for k,v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            # batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
            loss = outputs.loss
            train_loss += loss.detach().float()
    loss = train_loss.float()
    # average loss over the number of examples
    return float(loss) / total_num

def get_score(weights, model, cache, data_module, batch_size, get_loss, get_regular, args):
    # the composed lora state dict
    final_state_dict = {}
    # all keys are the same
    keys = cache[0].keys()
    for i in range(len(cache)):
        lora_state_dict = cache[i]
        if i == 0:
            for key in keys:
                final_state_dict[key] = weights[i] * lora_state_dict[key]
        else:
            for key in keys:
                final_state_dict[key] = (
                    final_state_dict[key] + weights[i] * lora_state_dict[key]
                )
    with torch.no_grad():
        if 'zero3' in args.deepspeed:
            load_deepspeed(final_state_dict, model, strict=False)
        else:
            model.load_state_dict(final_state_dict, strict=False)  
        
    # minimize the metric
    loss = get_loss(data_module, model, batch_size)
    # L1 regularization term
    metric_val = loss + get_regular(weights)
    
    return metric_val

def get_regular(weights):
    """
    Get the L1 regularization term for the weights
    """
    sum_of_squares = sum([abs(x) for x in weights]) / len(weights)
    return 0.05 * sum_of_squares

def fdlora_aggregate_state_dict(global_state_dict_list, local_state_dict_list, selected_ids, num_selection, training_args, **kwargs):
    # only aggregate the local models with the same architecture
    model_ids = kwargs['model_ids']
    curr_round = kwargs['curr_round']
    models = kwargs['models']
    
    for model_id, homo_client_ids in model_ids.items():
        global_state_dict = global_state_dict_list[homo_client_ids[0]]
        
        # only use active clients
        active_homo_ids = [id for id in homo_client_ids if id in selected_ids]
        
        for key in global_state_dict.keys():
            global_state_dict[key] = sum([local_state_dict_list[client][key] / len(active_homo_ids) for client in active_homo_ids])
        for i in homo_client_ids:
            global_state_dict_list[i] = global_state_dict

    # combine local and global and save it
    data_path = "dataset/llava_finetune/llava_v1_5_mix665k_mixed.json"
    # data_path = 'chatbotIT.json'
    public_datalist = json.load(open(data_path, "r"))
    random.shuffle(public_datalist)
    data_module = make_supervised_data_module(client_data=public_datalist[:12], # sub_dataset
                                                tokenizer=kwargs['tokenizer'],
                                                processor=kwargs['processor'],
                                                data_args=copy.deepcopy(kwargs['data_args']))
    
    for client_id in selected_ids:
        cache = [local_state_dict_list[client_id], global_state_dict_list[client_id]]

        model_id = None
        for mid, homo_ids in model_ids.items():
            if client_id in homo_ids:
                model_id = mid
        model = models[model_id]
        model = model.cuda()
        get_score_partial = partial(get_score, 
                                    model=model, 
                                    cache=cache,
                                    data_module=data_module,
                                    batch_size=4,
                                    get_loss=get_loss, 
                                    get_regular=get_regular,
                                    args=training_args)
        # set up the limit of the weights
        instrum = ng.p.Array(
            init=[0] * 2,
            upper=[1.5] * 2,
            lower=[-1.5] * 2,
        )
        optimizer = ng.optimizers.NGOpt(parametrization=instrum, budget=40)
        print("> Begin to perform gradient-free optimization ...")
        recommendation = optimizer.minimize(get_score_partial, verbosity=1)
        final_lora = get_final_weights(recommendation.value, cache)
        
        output_dir = os.path.join(training_args.state_dir, f"{client_id}_client_model_round{curr_round+1}.pth")
        torch.save(final_lora, output_dir)
        
        model = model.cpu()

def fdlora_blockwise_aggregate_state_dict(global_state_dict_list, local_state_dict_list, selected_ids, num_selection, training_args, **kwargs):
    # only aggregate the local models with the same architecture
    model_ids = kwargs['model_ids']
    curr_round = kwargs['curr_round']
    models = kwargs['models']
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

    # combine local and global and save it
    data_path = "dataset/llava_finetune/llava_v1_5_mix665k_mixed.json"
    # data_path = 'chatbotIT.json'
    public_datalist = json.load(open(data_path, "r"))
    random.shuffle(public_datalist)
    data_module = make_supervised_data_module(client_data=public_datalist[:12], # sub_dataset
                                                tokenizer=kwargs['tokenizer'],
                                                processor=kwargs['processor'],
                                                data_args=copy.deepcopy(kwargs['data_args']))
    
    for client_id in selected_ids:
        cache = [local_state_dict_list[client_id], global_state_dict_list[client_id]]

        model_id = None
        for mid, homo_ids in model_ids.items():
            if client_id in homo_ids:
                model_id = mid
        model = models[model_id]
        model = model.cuda()
        get_score_partial = partial(get_score, 
                                    model=model, 
                                    cache=cache,
                                    data_module=data_module,
                                    batch_size=4,
                                    get_loss=get_loss, 
                                    get_regular=get_regular,
                                    args=training_args)
        # set up the limit of the weights
        instrum = ng.p.Array(
            init=[0] * 2,
            upper=[1.5] * 2,
            lower=[-1.5] * 2,
        )
        optimizer = ng.optimizers.NGOpt(parametrization=instrum, budget=40)
        print("> Begin to perform gradient-free optimization ...")
        recommendation = optimizer.minimize(get_score_partial, verbosity=1)
        final_lora = get_final_weights(recommendation.value, cache)
        
        output_dir = os.path.join(training_args.state_dir, f"{client_id}_client_model_round{curr_round+1}.pth")
        torch.save(final_lora, output_dir)
        
        model = model.cpu()
