import torch
from utils.train_utils import load_deepspeed
import copy

def fedmosaic_load_state_dict(model, global_state_dict, local_state_dict_list, client_id, training_args, extra_state_dict_dict=None):
    # first load loca model and then load global model
    with torch.no_grad():
        if 'zero3' in training_args.deepspeed:
            load_deepspeed(local_state_dict_list[client_id], model, strict=False)
        else:
            model.load_state_dict(local_state_dict_list[client_id], strict=False)  
            
        # gradient based similarity wegithed averaging (exclude own)
        if extra_state_dict_dict['curr_round'] > 0 and 'task_similarity' in extra_state_dict_dict:
            # active clients
            active_clients = extra_state_dict_dict['selected_ids']
            
            # similarity matrix
            sim = extra_state_dict_dict['task_similarity']
            new_global_state_dict = {}
            
            weights = sim[client_id].clone()
            
            weights[client_id] = -1e9
            weights = (weights/training_args.softmax_temp).softmax(dim=0)
            
            sim_sum = weights.sum() - weights[client_id]
            
            for name in global_state_dict.keys():
                new_param = 0
                if 'lora1' in name:
                    target_key = name.replace('lora1', 'lora2')
                else:
                    new_global_state_dict[name] = local_state_dict_list[client_id][name]
                    continue
                
                for id in active_clients:
                    if id == client_id:
                        continue
                    # if training_args.is_hetero_model:
                    #     breakpoint()
                    # else:
                    new_param += weights[id]*local_state_dict_list[id][target_key] / sim_sum
                if isinstance(new_param, int):
                    continue
                new_global_state_dict[name] = new_param
        else:
            new_global_state_dict = global_state_dict
        if 'zero3' in training_args.deepspeed:
            load_deepspeed(new_global_state_dict, model, strict=False)
        else:
            model.load_state_dict(new_global_state_dict, strict=False) 


def fedmosaic_2block_load_state_dict(model, global_state_dict, local_state_dict_list, client_id, training_args, extra_state_dict_dict):
    # first load loca model and then load global model
    layer_index = extra_state_dict_dict['LAYER_INDEX']
    with torch.no_grad():
        if 'zero3' in training_args.deepspeed:
            load_deepspeed(local_state_dict_list[client_id], model, strict=False)
        else:
            model.load_state_dict(local_state_dict_list[client_id], strict=False)
            
        model_ids = extra_state_dict_dict['model_ids']
        
        for model_id, homo_ids in model_ids.items():
            if client_id in homo_ids:
                homo_client_ids = homo_ids    
        
        new_global_state_dict = {}
        for key in local_state_dict_list[client_id].keys():
            if 'lora2' in key:
                new_key = key.replace('lora2','lora1')
                new_global_state_dict[new_key] = copy.deepcopy(local_state_dict_list[client_id][key])

        # gradient based similarity wegithed averaging (exclude own)
        if extra_state_dict_dict['curr_round'] > 0 and 'task_similarity' in extra_state_dict_dict:
            # similarity matrix
            sim = extra_state_dict_dict['task_similarity']
            
            weights = sim[client_id].clone()
            homo_weights = sim[client_id].clone()
            
            weights[client_id] = -1e9
            weights = (weights/training_args.softmax_temp).softmax(dim=0)
            sim_sum = weights.sum() - weights[client_id]
            
            for id in range(training_args.num_clients):
                if id not in homo_client_ids:
                    homo_weights[id] = -1e9
            
            homo_weights[client_id] = -1e9
            homo_weights = (homo_weights/training_args.softmax_temp).softmax(dim=0)
            
            homo_sim_sum = homo_weights.sum() - homo_weights[client_id]
            
            
            # # weights[client_id] = sim_sum
            # # sim_sum += sim_sum
            cur_layer_num = []
            for k in global_state_dict.keys():
                if 'layers.' in k:
                    cur_layer_num.append(int(k.split('.')[layer_index]))
            cur_layer_num = sorted(list(set(cur_layer_num)))
            cur_layer_num = len(set(cur_layer_num)) // 2
            cur_layer_num = [cur_layer_num*1 -1,cur_layer_num*2 -1]
            for name in global_state_dict.keys():
                new_param = 0
                if 'lora1' in name:
                    target_key = name.replace('lora1', 'lora2')
                else:
                    new_global_state_dict[name] = local_state_dict_list[client_id][name]
                    continue
                
                splited = target_key.split('.')
                if int(splited[layer_index]) in cur_layer_num:
                    if 'lora2_P' not in target_key and 'lora2_Q' not in target_key:
                        continue
                    
                    for id in range(training_args.num_clients):
                        if id == client_id:
                            continue
                        else:
                            splited = target_key.split('.')
                            # if layer number is different
                            layer_num = []
                            for k in local_state_dict_list[id].keys():
                                if 'layers.' in k:
                                    layer_num.append(int(k.split('.')[layer_index]))
                            layer_num = len(set(layer_num)) // 2
                            
                            target_layers = [layer_num*1 -1,layer_num*2 -1]
                            if cur_layer_num[-1] != target_layers[-1]: # if different size
                                if int(splited[layer_index]) == cur_layer_num[0]: # mid layer
                                    splited[layer_index] = str(target_layers[0])
                                elif int(splited[layer_index]) == cur_layer_num[1]: # last layer 
                                    splited[layer_index] = str(target_layers[1])
                                new_target_key = '.'.join(splited)
                            else:
                                new_target_key = target_key
                            new_param += weights[id]*local_state_dict_list[id][new_target_key] / sim_sum
                else:
                    for id in homo_client_ids:
                        if id == client_id:
                            continue
                        new_param += homo_weights[id]*local_state_dict_list[id][target_key] / homo_sim_sum
                
                if isinstance(new_param, int):
                    continue
                new_global_state_dict[name] = new_param
        if 'zero3' in training_args.deepspeed:
            load_deepspeed(new_global_state_dict, model, strict=False)
        else:
            model.load_state_dict(new_global_state_dict, strict=False) 

def fedmosaic_4block_load_state_dict(model, global_state_dict, local_state_dict_list, client_id, training_args, extra_state_dict_dict):
    # first load loca model and then load global model
    layer_index = extra_state_dict_dict['LAYER_INDEX']
    with torch.no_grad():
        if 'zero3' in training_args.deepspeed:
            load_deepspeed(local_state_dict_list[client_id], model, strict=False)
        else:
            model.load_state_dict(local_state_dict_list[client_id], strict=False)
            
        model_ids = extra_state_dict_dict['model_ids']
        
        for model_id, homo_ids in model_ids.items():
            if client_id in homo_ids:
                homo_client_ids = homo_ids    
        
        new_global_state_dict = {}
        for key in local_state_dict_list[client_id].keys():
            if 'lora2' in key:
                new_key = key.replace('lora2','lora1')
                new_global_state_dict[new_key] = copy.deepcopy(local_state_dict_list[client_id][key])

        # gradient based similarity wegithed averaging (exclude own)
        if extra_state_dict_dict['curr_round'] > 0 and 'task_similarity' in extra_state_dict_dict:
            # similarity matrix
            sim = extra_state_dict_dict['task_similarity']
            
            weights = sim[client_id].clone()
            homo_weights = sim[client_id].clone()
            
            weights[client_id] = -1e9
            weights = (weights/training_args.softmax_temp).softmax(dim=0)
            sim_sum = weights.sum() - weights[client_id]
            
            for id in range(training_args.num_clients):
                if id not in homo_client_ids:
                    homo_weights[id] = -1e9
            
            homo_weights[client_id] = -1e9
            homo_weights = (homo_weights/training_args.softmax_temp).softmax(dim=0)
            
            homo_sim_sum = homo_weights.sum() - homo_weights[client_id]
            
            
            # # weights[client_id] = sim_sum
            # # sim_sum += sim_sum
            cur_layer_num = []
            for k in global_state_dict.keys():
                if 'layers.' in k:
                    cur_layer_num.append(int(k.split('.')[layer_index]))
            cur_layer_num = sorted(list(set(cur_layer_num)))
            cur_layer_num = len(set(cur_layer_num)) // 4     
            cur_layer_num = [cur_layer_num*1 -1,cur_layer_num*2 -1,cur_layer_num*3 -1,cur_layer_num*4 -1]
            for name in global_state_dict.keys():
                new_param = 0
                if 'lora1' in name:
                    target_key = name.replace('lora1', 'lora2')
                else:
                    new_global_state_dict[name] = local_state_dict_list[client_id][name]
                    continue
                
                splited = target_key.split('.')
                if int(splited[layer_index]) in cur_layer_num:
                    if 'lora2_P' not in target_key and 'lora2_Q' not in target_key:
                        continue
                    
                    for id in range(training_args.num_clients):
                        if id == client_id:
                            continue
                        else:
                            splited = target_key.split('.')
                            # if layer number is different
                            layer_num = []
                            for k in local_state_dict_list[id].keys():
                                if 'layers.' in k:
                                    layer_num.append(int(k.split('.')[layer_index]))
                            layer_num = len(set(layer_num)) // 4
                            
                            target_layers = [layer_num*1 -1,layer_num*2 -1,layer_num*3 -1,layer_num*4 -1]
                            if cur_layer_num[-1] != target_layers[-1]: # if different size
                                index = cur_layer_num.index(int(splited[layer_index]))
                                splited[layer_index] = str(target_layers[index])
                                new_target_key = '.'.join(splited)
                            else:
                                new_target_key = target_key
                            new_param += weights[id]*local_state_dict_list[id][new_target_key] / sim_sum
                else:
                    for id in homo_client_ids:
                        if id == client_id:
                            continue
                        new_param += homo_weights[id]*local_state_dict_list[id][target_key] / homo_sim_sum
                if isinstance(new_param, int):
                    continue
                new_global_state_dict[name] = new_param

        if 'zero3' in training_args.deepspeed:
            load_deepspeed(new_global_state_dict, model, strict=False)
        else:
            model.load_state_dict(new_global_state_dict, strict=False) 


def fedmosaic_8block_load_state_dict(model, global_state_dict, local_state_dict_list, client_id, training_args, extra_state_dict_dict):
    layer_index = extra_state_dict_dict['LAYER_INDEX']
    # first load loca model and then load global model
    with torch.no_grad():
        if 'zero3' in training_args.deepspeed:
            load_deepspeed(local_state_dict_list[client_id], model, strict=False)
        else:
            model.load_state_dict(local_state_dict_list[client_id], strict=False)
        
        model_ids = extra_state_dict_dict['model_ids']
        
        for model_id, homo_ids in model_ids.items():
            if client_id in homo_ids:
                homo_client_ids = homo_ids  
        
        new_global_state_dict = {}
        for key in local_state_dict_list[client_id].keys():
            if 'lora2' in key:
                new_key = key.replace('lora2','lora1')
                new_global_state_dict[new_key] = copy.deepcopy(local_state_dict_list[client_id][key])

        # gradient based similarity wegithed averaging (exclude own)
        if extra_state_dict_dict['curr_round'] > 0 and 'task_similarity' in extra_state_dict_dict:  
            # similarity matrix
            sim = extra_state_dict_dict['task_similarity']
            
            weights = sim[client_id].clone()
            homo_weights = sim[client_id].clone()
            
            weights[client_id] = -1e9
            weights = (weights/training_args.softmax_temp).softmax(dim=0)
            sim_sum = weights.sum() - weights[client_id]
            
            for id in range(training_args.num_clients):
                if id not in homo_client_ids:
                    homo_weights[id] = -1e9
            
            homo_weights[client_id] = -1e9
            homo_weights = (homo_weights/training_args.softmax_temp).softmax(dim=0)
            
            homo_sim_sum = homo_weights.sum() - homo_weights[client_id]
            
            cur_layer_num = []
            for k in global_state_dict.keys():
                if 'layers.' in k:
                    cur_layer_num.append(int(k.split('.')[layer_index]))
            cur_layer_num = len(list(set(cur_layer_num)))
            if layer_index == 5: # multimodal model
                if cur_layer_num == 16:
                    cur_layer_num = [1,3,5,7,9,11,13,15] # llama3.2 1B
                elif cur_layer_num == 24:
                    cur_layer_num = [2,5,8,11,14,17,20,23] # qwen2.5 0.5B
                elif cur_layer_num == 28:
                    cur_layer_num = [2,5,8,11,14,17,20,27] # llama3.2 3B, qwen2.5 1.5B
                elif cur_layer_num == 32:
                    cur_layer_num = [3,7,11,15,19,23,27,31] # llama3.1 8B
                elif cur_layer_num == 36:
                    cur_layer_num = [3,7,11,15,19,23,27,35] # qwen2.5 3B
            
            for name in global_state_dict.keys():
                new_param = 0
                if 'lora1' in name:
                    target_key = name.replace('lora1', 'lora2')
                
                splited = target_key.split('.')
                if int(splited[layer_index]) in cur_layer_num:
                    if 'lora2_P' not in target_key and 'lora2_Q' not in target_key:
                        continue
                
                    for id in range(training_args.num_clients):
                        if id == client_id:
                            continue
                        else:
                            splited = target_key.split('.')
                            # if layer number is different
                            layer_num = []
                            for k in local_state_dict_list[id].keys():
                                if 'layers.' in k:
                                    layer_num.append(int(k.split('.')[layer_index]))
                            layer_num = len(set(layer_num))
                            
                            if layer_index == 5: # multimodal model
                                if layer_num == 16:
                                    target_layers = [1,3,5,7,9,11,13,15] # llama3.2 1B
                                elif layer_num == 24:
                                    target_layers = [2,5,8,11,14,17,20,23] # qwen2.5 0.5B
                                elif layer_num == 28:
                                    target_layers = [2,5,8,11,14,17,20,27] # llama3.2 3B, qwen2.5 1.5B
                                elif layer_num == 32:
                                    target_layers = [3,7,11,15,19,23,27,31] # llama3.1 8B
                                elif layer_num == 36:
                                    target_layers = [3,7,11,15,19,23,27,35] # qwen2.5 3B
                            
                            if cur_layer_num[-1] != target_layers[-1]: # if different size
                                index = cur_layer_num.index(int(splited[layer_index]))
                                splited[layer_index] = str(target_layers[index])
                                new_target_key = '.'.join(splited)
                            else:
                                new_target_key = target_key
                            new_param += weights[id]*local_state_dict_list[id][new_target_key] / sim_sum
                else:
                    for id in homo_client_ids:
                        if id == client_id:
                            continue
                        new_param += homo_weights[id]*local_state_dict_list[id][target_key] / homo_sim_sum
                if isinstance(new_param, int):
                    continue
                new_global_state_dict[name] = new_param
        if 'zero3' in training_args.deepspeed:
            load_deepspeed(new_global_state_dict, model, strict=False)
        else:
            model.load_state_dict(new_global_state_dict, strict=False) 


def feddualMultipq_homoAgg_t5_load_state_dict(model, global_state_dict, local_state_dict_list, client_id, training_args, extra_state_dict_dict):
    # first load loca model and then load global model
    layer_index = extra_state_dict_dict['LAYER_INDEX']
    with torch.no_grad():
        if 'zero3' in training_args.deepspeed:
            load_deepspeed(local_state_dict_list[client_id], model, strict=False)
        else:
            model.load_state_dict(local_state_dict_list[client_id], strict=False)
            
        model_ids = extra_state_dict_dict['model_ids']
        
        for model_id, homo_ids in model_ids.items():
            if client_id in homo_ids:
                homo_client_ids = homo_ids    
        
        new_global_state_dict = {}
        for key in local_state_dict_list[client_id].keys():
            if 'lora2' in key:
                new_key = key.replace('lora2','lora1')
                new_global_state_dict[new_key] = copy.deepcopy(local_state_dict_list[client_id][key])

        # gradient based similarity wegithed averaging (exclude own)
        if extra_state_dict_dict['curr_round'] > 0 and 'task_similarity' in extra_state_dict_dict:
            # similarity matrix
            sim = extra_state_dict_dict['task_similarity']
            
            weights = sim[client_id].clone()
            homo_weights = sim[client_id].clone()
            
            weights[client_id] = -1e9
            weights = (weights/training_args.softmax_temp).softmax(dim=0)
            sim_sum = weights.sum() - weights[client_id]
            
            for id in range(training_args.num_clients):
                if id not in homo_client_ids:
                    homo_weights[id] = -1e9
            
            homo_weights[client_id] = -1e9
            homo_weights = (homo_weights/training_args.softmax_temp).softmax(dim=0)
            
            homo_sim_sum = homo_weights.sum() - homo_weights[client_id]
            
            
            # # weights[client_id] = sim_sum
            # # sim_sum += sim_sum
            cur_layer_num = []
            for k in global_state_dict.keys():
                if 'layers.' in k:
                    cur_layer_num.append(int(k.split('.')[layer_index]))
            cur_layer_num = sorted(list(set(cur_layer_num)))
            cur_layer_num = len(set(cur_layer_num)) // 2
            cur_layer_num = [cur_layer_num*1 -1,cur_layer_num*2 -1]
            for name in global_state_dict.keys():
                new_param = 0
                if 'lora1' in name:
                    target_key = name.replace('lora1', 'lora2')
                elif 'ia3_l_1' in name:
                    target_key = name.replace('ia3_l_1', 'ia3_l_2')
                else: # lora3 or lora4
                    if training_args.share_ema:
                        target_key = name
                    else:
                        new_global_state_dict[name] = local_state_dict_list[client_id][name]
                        continue
                
                splited = target_key.split('.')
                if int(splited[layer_index]) in cur_layer_num:
                    if 'lora2_P' not in target_key and 'lora2_Q' not in target_key:
                        continue
                    
                    for id in range(training_args.num_clients):
                        if id == client_id:
                            continue
                        else:
                            splited = target_key.split('.')
                            # if layer number is different
                            layer_num = []
                            for k in local_state_dict_list[id].keys():
                                if 'layers.' in k:
                                    layer_num.append(int(k.split('.')[layer_index]))
                            layer_num = len(set(layer_num)) // 2
                            
                            target_layers = [layer_num*1 -1,layer_num*2 -1]
                            if cur_layer_num[-1] != target_layers[-1]: # if different size
                                index = cur_layer_num.index(int(splited[layer_index]))
                                splited[layer_index] = str(target_layers[index])
                                new_target_key = '.'.join(splited)
                            else:
                                new_target_key = target_key
                            new_param += weights[id]*local_state_dict_list[id][new_target_key] / sim_sum
                else:
                    for id in homo_client_ids:
                        if id == client_id:
                            continue
                        new_param += homo_weights[id]*local_state_dict_list[id][target_key] / homo_sim_sum
                if isinstance(new_param, int):
                    continue
                new_global_state_dict[name] = new_param
            # if (training_args.local_rank == 0 or training_args.local_rank == -1):
            #     output_dir = os.path.join(training_args.state_dir, f"{client_id}_client_global_model_round{extra_state_dict_dict['curr_round']}.pth")
            #     torch.save(new_global_state_dict, output_dir)
        # else:
        #     new_global_state_dict = global_state_dict
        if 'zero3' in training_args.deepspeed:
            load_deepspeed(new_global_state_dict, model, strict=False)
        else:
            model.load_state_dict(new_global_state_dict, strict=False) 
