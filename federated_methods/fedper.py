import torch
from utils.train_utils import load_deepspeed

def fedper_set_state_dict(model, global_state_dict, local_state_dict_list, training_args):
    layer_num = []
    for k in global_state_dict.keys():
        if 'layers.' in k:
            layer_num.append(int(k.split('.')[4]))
    layer_num = sorted(list(set(layer_num)))
    
    layers_to_del = layer_num[-1:]
    
    keys_to_del = []
    for k in global_state_dict.keys():
        if 'layers.' in k and int(k.split('.')[4]) in layers_to_del:
            keys_to_del.append(k)
    for k in keys_to_del:
        del global_state_dict[k]
    return {}

def fedper_load_state_dict(model, global_state_dict, local_state_dict_list, client_id, training_args, extra_state_dict_dict):
    # first load loca model and then load global model
    with torch.no_grad():
        if 'zero3' in training_args.deepspeed:
            load_deepspeed(local_state_dict_list[client_id], model, strict=False)
        else:
            model.load_state_dict(local_state_dict_list[client_id], strict=False)    
        if 'zero3' in training_args.deepspeed:
            load_deepspeed(global_state_dict, model, strict=False)
        else:
            model.load_state_dict(global_state_dict, strict=False) 