import torch
from utils.train_utils import load_deepspeed
import copy

def fedpq_load_state_dict(model, global_state_dict, local_state_dict_list, client_id, training_args, extra_state_dict_dict):
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
            

def fedlastpq_load_state_dict(model, global_state_dict, local_state_dict_list, client_id, training_args, extra_state_dict_dict):
    # first load loca model and then load global model
    with torch.no_grad():
        if 'zero3' in training_args.deepspeed:
            load_deepspeed(local_state_dict_list[client_id], model, strict=False)
        else:
            model.load_state_dict(local_state_dict_list[client_id], strict=False)    
        
        new_global_state_dict = {}
        for name in global_state_dict.keys():
            new_param = 0
            target_key = name
            
            for id in range(training_args.num_clients):
                # if layer number is different
                layer_num = []
                for k in local_state_dict_list[id].keys():
                    if 'layers.' in k:
                        layer_num.append(int(k.split('.')[5]))
                layer_num = sorted(list(set(layer_num)))
                splited = target_key.split('.')
                if int(splited[5]) != layer_num[-1]: # last layer
                    splited[5] = str(layer_num[-1])
                    target_key = '.'.join(splited)
                new_param += local_state_dict_list[id][target_key] / training_args.num_clients
                
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

def fedlastpq_tv_load_state_dict(model, global_state_dict, local_state_dict_list, client_id, training_args, extra_state_dict_dict):
    # first load loca model and then load global model
    with torch.no_grad():
        if 'zero3' in training_args.deepspeed:
            load_deepspeed(local_state_dict_list[client_id], model, strict=False)
        else:
            model.load_state_dict(local_state_dict_list[client_id], strict=False)    
        
        # gradient based similarity wegithed averaging (exclude own)
        if extra_state_dict_dict['curr_round'] > 0 and 'task_similarity' in extra_state_dict_dict:
            new_global_state_dict = {}
            # similarity matrix
            sim = extra_state_dict_dict['task_similarity']
            
            weights = sim[client_id].clone()
            
            # weights[client_id] = -1e9
            weights = (weights).softmax(dim=0)
            
            sim_sum = weights.sum() #- weights[client_id]
            
            # # weights[client_id] = sim_sum
            # # sim_sum += sim_sum
            
            for name in global_state_dict.keys():
                new_param = 0
                for id in range(training_args.num_clients):
                    # if layer number is different
                    layer_num = []
                    for k in local_state_dict_list[id].keys():
                        if 'layers.' in k:
                            layer_num.append(int(k.split('.')[5]))
                    layer_num = sorted(list(set(layer_num)))
                    splited = name.split('.')
                    if int(splited[5]) != layer_num[-1]: # last layer
                        splited[5] = str(layer_num[-1])
                        target_key = '.'.join(splited)
                    else:
                        target_key = name
                    new_param += weights[id]*local_state_dict_list[id][target_key] / sim_sum
                    
                new_global_state_dict[name] = new_param
        else:
            new_global_state_dict = global_state_dict
        if 'zero3' in training_args.deepspeed:
            load_deepspeed(new_global_state_dict, model, strict=False)
        else:
            model.load_state_dict(new_global_state_dict, strict=False) 


def fedFLpq_load_state_dict(model, global_state_dict, local_state_dict_list, client_id, training_args, extra_state_dict_dict):
    # first load loca model and then load global model
    with torch.no_grad():
        if 'zero3' in training_args.deepspeed:
            load_deepspeed(local_state_dict_list[client_id], model, strict=False)
        else:
            model.load_state_dict(local_state_dict_list[client_id], strict=False)    
        
        new_global_state_dict = {}
        for name in global_state_dict.keys():
            new_param = 0
            target_key = name
            
            for id in range(training_args.num_clients):
                splited = target_key.split('.')
                if int(splited[5]) != 0: # first layer
                    # if layer number is different
                    layer_num = []
                    for k in local_state_dict_list[id].keys():
                        if 'layers.' in k:
                            layer_num.append(int(k.split('.')[5]))
                    layer_num = sorted(list(set(layer_num)))
                    
                    if int(splited[5]) != layer_num[-1]: # last layer
                        splited[5] = str(layer_num[-1])
                        target_key = '.'.join(splited)
                new_param += local_state_dict_list[id][target_key] / training_args.num_clients
                
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

def fedFMLpq_load_state_dict(model, global_state_dict, local_state_dict_list, client_id, training_args, extra_state_dict_dict):
    # first load loca model and then load global model
    with torch.no_grad():
        if 'zero3' in training_args.deepspeed:
            load_deepspeed(local_state_dict_list[client_id], model, strict=False)
        else:
            model.load_state_dict(local_state_dict_list[client_id], strict=False)    
        
        new_global_state_dict = {}
        
        cur_layer_num = []
        for k in global_state_dict.keys():
            if 'layers.' in k:
                cur_layer_num.append(int(k.split('.')[5]))
        cur_layer_num = sorted(list(set(cur_layer_num)))
        
        for name in global_state_dict.keys():
            new_param = 0
            target_key = name
            
            for id in range(training_args.num_clients):
                splited = target_key.split('.')
                if int(splited[5]) != 0: # first layer
                    # if layer number is different
                    layer_num = []
                    for k in local_state_dict_list[id].keys():
                        if 'layers.' in k:
                            layer_num.append(int(k.split('.')[5]))
                    layer_num = sorted(list(set(layer_num)))
                    
                    mid_layer = layer_num[int(len(layer_num)/2)] - 1
                    
                    if cur_layer_num[-1] != layer_num[-1]: # if different size
                        if int(splited[5]) == cur_layer_num[-2]: # mid layer
                            splited[5] = str(mid_layer)
                        elif int(splited[5]) == cur_layer_num[-1]: # last layer 
                            splited[5] = str(layer_num[-1])
                        new_target_key = '.'.join(splited)
                    else:
                        new_target_key = target_key
                else: 
                    new_target_key = target_key
                new_param += local_state_dict_list[id][new_target_key] / training_args.num_clients
                
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

def fedMultipq_load_state_dict(model, global_state_dict, local_state_dict_list, client_id, training_args, extra_state_dict_dict):
    # first load loca model and then load global model
    with torch.no_grad():
        if 'zero3' in training_args.deepspeed:
            load_deepspeed(local_state_dict_list[client_id], model, strict=False)
        else:
            model.load_state_dict(local_state_dict_list[client_id], strict=False)    
        
        cur_layer_num = []
        for k in global_state_dict.keys():
            if 'layers.' in k:
                cur_layer_num.append(int(k.split('.')[5]))
        cur_layer_num = sorted(list(set(cur_layer_num)))
        new_global_state_dict = {}
        for name in global_state_dict.keys():
            new_param = 0
            target_key = name
            
            for id in range(training_args.num_clients):
                # if layer number is different
                splited = target_key.split('.')
                # if layer number is different
                layer_num = []
                for k in local_state_dict_list[id].keys():
                    if 'layers.' in k:
                        layer_num.append(int(k.split('.')[5]))
                layer_num = len(set(layer_num)) // 4
                
                target_layers = [layer_num*1 -1,layer_num*2 -1,layer_num*3 -1,layer_num*4 -1]
                if cur_layer_num[-1] != target_layers[-1]: # if different size
                    if int(splited[5]) == cur_layer_num[0]: # mid layer
                        splited[5] = str(target_layers[0])
                    elif int(splited[5]) == cur_layer_num[1]: # last layer 
                        splited[5] = str(target_layers[1])
                    elif int(splited[5]) == cur_layer_num[2]: # last layer 
                        splited[5] = str(target_layers[2])
                    elif int(splited[5]) == cur_layer_num[3]: # last layer 
                        splited[5] = str(target_layers[3])
                    new_target_key = '.'.join(splited)
                else:
                    new_target_key = target_key
                new_param += local_state_dict_list[id][new_target_key] / training_args.num_clients
                
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

def fedMultipq_tv_load_state_dict(model, global_state_dict, local_state_dict_list, client_id, training_args, extra_state_dict_dict):
    # first load loca model and then load global model
    with torch.no_grad():
        if 'zero3' in training_args.deepspeed:
            load_deepspeed(local_state_dict_list[client_id], model, strict=False)
        else:
            model.load_state_dict(local_state_dict_list[client_id], strict=False)    
        
        # gradient based similarity wegithed averaging (exclude own)
        if extra_state_dict_dict['curr_round'] > 0 and 'task_similarity' in extra_state_dict_dict:
            # similarity matrix
            sim = extra_state_dict_dict['task_similarity']
            
            weights = sim[client_id].clone()
            
            # weights[client_id] = -1e9
            weights = (weights).softmax(dim=0)
            
            sim_sum = weights.sum() #- weights[client_id]
            
            # # weights[client_id] = sim_sum
            # # sim_sum += sim_sum
            cur_layer_num = []
            for k in global_state_dict.keys():
                if 'layers.' in k:
                    cur_layer_num.append(int(k.split('.')[5]))
            cur_layer_num = sorted(list(set(cur_layer_num)))
            new_global_state_dict = {}
            for name in global_state_dict.keys():
                new_param = 0
                for id in range(training_args.num_clients):
                    # if layer number is different
                    splited = name.split('.')
                    layer_num = []
                    for k in local_state_dict_list[id].keys():
                        if 'layers.' in k:
                            layer_num.append(int(k.split('.')[5]))
                    layer_num = len(set(layer_num)) // 4
                
                    target_layers = [layer_num*1 -1,layer_num*2 -1,layer_num*3 -1,layer_num*4 -1]
                    if cur_layer_num[-1] != target_layers[-1]: # if different size
                        if int(splited[5]) == cur_layer_num[0]: # mid layer
                            splited[5] = str(target_layers[0])
                        elif int(splited[5]) == cur_layer_num[1]: # last layer 
                            splited[5] = str(target_layers[1])
                        elif int(splited[5]) == cur_layer_num[2]: # last layer 
                            splited[5] = str(target_layers[2])
                        elif int(splited[5]) == cur_layer_num[3]: # last layer 
                            splited[5] = str(target_layers[3])
                        new_target_key = '.'.join(splited)
                    else:
                        new_target_key = name
                    new_param += weights[id]*local_state_dict_list[id][new_target_key] / sim_sum
                    
                new_global_state_dict[name] = new_param
        else:
            new_global_state_dict = global_state_dict
        if 'zero3' in training_args.deepspeed:
            load_deepspeed(new_global_state_dict, model, strict=False)
        else:
            model.load_state_dict(new_global_state_dict, strict=False) 

def fedMulti2pq_load_state_dict(model, global_state_dict, local_state_dict_list, client_id, training_args, extra_state_dict_dict):
    # first load loca model and then load global model
    with torch.no_grad():
        if 'zero3' in training_args.deepspeed:
            load_deepspeed(local_state_dict_list[client_id], model, strict=False)
        else:
            model.load_state_dict(local_state_dict_list[client_id], strict=False)    
        
        cur_layer_num = []
        for k in global_state_dict.keys():
            if 'layers.' in k:
                cur_layer_num.append(int(k.split('.')[5]))
        cur_layer_num = sorted(list(set(cur_layer_num)))
        new_global_state_dict = {}
        for name in global_state_dict.keys():
            new_param = 0
            target_key = name
            
            for id in range(training_args.num_clients):
                # if layer number is different
                splited = target_key.split('.')
                # if layer number is different
                layer_num = []
                for k in local_state_dict_list[id].keys():
                    if 'layers.' in k:
                        layer_num.append(int(k.split('.')[5]))
                layer_num = len(set(layer_num)) // 4
                
                target_layers = [layer_num*1 -2, layer_num*1 -1, layer_num*2 -2, layer_num*2 -1,
                                 layer_num*3 -2, layer_num*3 -1, layer_num*4 -2, layer_num*4 -1]
                if cur_layer_num[-1] != target_layers[-1]: # if different size
                    if int(splited[5]) == cur_layer_num[0]: # mid layer
                        splited[5] = str(target_layers[0])
                    elif int(splited[5]) == cur_layer_num[1]: # last layer 
                        splited[5] = str(target_layers[1])
                    elif int(splited[5]) == cur_layer_num[2]: # last layer 
                        splited[5] = str(target_layers[2])
                    elif int(splited[5]) == cur_layer_num[3]: # last layer 
                        splited[5] = str(target_layers[3])
                    elif int(splited[5]) == cur_layer_num[4]: # last layer 
                        splited[5] = str(target_layers[4])
                    elif int(splited[5]) == cur_layer_num[5]: # last layer 
                        splited[5] = str(target_layers[5])
                    elif int(splited[5]) == cur_layer_num[6]: # last layer 
                        splited[5] = str(target_layers[6])
                    elif int(splited[5]) == cur_layer_num[7]: # last layer 
                        splited[5] = str(target_layers[7])
                    new_target_key = '.'.join(splited)
                else:
                    new_target_key = target_key
                new_param += local_state_dict_list[id][new_target_key] / training_args.num_clients
                
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

def fedMulti2pq_tv_load_state_dict(model, global_state_dict, local_state_dict_list, client_id, training_args, extra_state_dict_dict):
    # first load loca model and then load global model
    with torch.no_grad():
        if 'zero3' in training_args.deepspeed:
            load_deepspeed(local_state_dict_list[client_id], model, strict=False)
        else:
            model.load_state_dict(local_state_dict_list[client_id], strict=False)    
        
        # gradient based similarity wegithed averaging (exclude own)
        if extra_state_dict_dict['curr_round'] > 0 and 'task_similarity' in extra_state_dict_dict:
            # similarity matrix
            sim = extra_state_dict_dict['task_similarity']
            
            weights = sim[client_id].clone()
            
            # weights[client_id] = -1e9
            weights = (weights).softmax(dim=0)
            
            sim_sum = weights.sum() #- weights[client_id]
            
            # # weights[client_id] = sim_sum
            # # sim_sum += sim_sum
            cur_layer_num = []
            for k in global_state_dict.keys():
                if 'layers.' in k:
                    cur_layer_num.append(int(k.split('.')[5]))
            cur_layer_num = sorted(list(set(cur_layer_num)))
            new_global_state_dict = {}
            for name in global_state_dict.keys():
                new_param = 0
                for id in range(training_args.num_clients):
                    # if layer number is different
                    splited = name.split('.')
                    layer_num = []
                    for k in local_state_dict_list[id].keys():
                        if 'layers.' in k:
                            layer_num.append(int(k.split('.')[5]))
                    layer_num = len(set(layer_num)) // 4
                
                    target_layers = [layer_num*1 -2, layer_num*1 -1, layer_num*2 -2, layer_num*2 -1,
                                     layer_num*3 -2, layer_num*3 -1, layer_num*4 -2, layer_num*4 -1]
                    if cur_layer_num[-1] != target_layers[-1]: # if different size
                        if int(splited[5]) == cur_layer_num[0]: # mid layer
                            splited[5] = str(target_layers[0])
                        elif int(splited[5]) == cur_layer_num[1]: # last layer 
                            splited[5] = str(target_layers[1])
                        elif int(splited[5]) == cur_layer_num[2]: # last layer 
                            splited[5] = str(target_layers[2])
                        elif int(splited[5]) == cur_layer_num[3]: # last layer 
                            splited[5] = str(target_layers[3])
                        elif int(splited[5]) == cur_layer_num[4]: # last layer 
                            splited[5] = str(target_layers[4])
                        elif int(splited[5]) == cur_layer_num[5]: # last layer 
                            splited[5] = str(target_layers[5])
                        elif int(splited[5]) == cur_layer_num[6]: # last layer 
                            splited[5] = str(target_layers[6])
                        elif int(splited[5]) == cur_layer_num[7]: # last layer 
                            splited[5] = str(target_layers[7])
                        new_target_key = '.'.join(splited)
                    else:
                        new_target_key = name
                    new_param += weights[id]*local_state_dict_list[id][new_target_key] / sim_sum
                    
                new_global_state_dict[name] = new_param
        else:
            new_global_state_dict = global_state_dict
        if 'zero3' in training_args.deepspeed:
            load_deepspeed(new_global_state_dict, model, strict=False)
        else:
            model.load_state_dict(new_global_state_dict, strict=False) 

def feddualpq_load_state_dict(model, global_state_dict, local_state_dict_list, client_id, training_args, extra_state_dict_dict):
    # first load loca model and then load global model
    with torch.no_grad():
        if 'zero3' in training_args.deepspeed:
            load_deepspeed(local_state_dict_list[client_id], model, strict=False)
        else:
            model.load_state_dict(local_state_dict_list[client_id], strict=False)
            
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
            
            weights[client_id] = -1e9
            weights = (weights/0.2).softmax(dim=0)
            
            sim_sum = weights.sum() - weights[client_id]
            
            # # weights[client_id] = sim_sum
            # # sim_sum += sim_sum
            
            for name in global_state_dict.keys():
                new_param = 0
                if 'lora1' in name:
                    target_key = name.replace('lora1', 'lora2')
                elif 'ia3_l_1' in name:
                    target_key = name.replace('ia3_l_1', 'ia3_l_2')
                
                for id in range(training_args.num_clients):
                    if id == client_id:
                        continue
                    if training_args.is_hetero_model:
                        breakpoint()
                    else:
                        new_param += weights[id]*local_state_dict_list[id][target_key] / sim_sum
                    
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

def fedduallastpq_load_state_dict(model, global_state_dict, local_state_dict_list, client_id, training_args, extra_state_dict_dict):
    # first load loca model and then load global model
    with torch.no_grad():
        if 'zero3' in training_args.deepspeed:
            load_deepspeed(local_state_dict_list[client_id], model, strict=False)
        else:
            model.load_state_dict(local_state_dict_list[client_id], strict=False)
            
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
            
            weights[client_id] = -1e9
            weights = (weights/0.2).softmax(dim=0)
            
            sim_sum = weights.sum() - weights[client_id]
            
            # # weights[client_id] = sim_sum
            # # sim_sum += sim_sum
            
            for name in global_state_dict.keys():
                new_param = 0
                if 'lora1' in name:
                    target_key = name.replace('lora1', 'lora2')
                elif 'ia3_l_1' in name:
                    target_key = name.replace('ia3_l_1', 'ia3_l_2')
                
                for id in range(training_args.num_clients):
                    if id == client_id:
                        continue
                    else:
                        # if layer number is different
                        layer_num = []
                        for k in local_state_dict_list[id].keys():
                            if 'layers.' in k:
                                layer_num.append(int(k.split('.')[5]))
                        layer_num = sorted(list(set(layer_num)))
                        splited = target_key.split('.')
                        if int(splited[5]) != layer_num[-1]: # last layer
                            splited[5] = str(layer_num[-1])
                            target_key = '.'.join(splited)
                        new_param += weights[id]*local_state_dict_list[id][target_key] / sim_sum
                    
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

def feddualFLpq_load_state_dict(model, global_state_dict, local_state_dict_list, client_id, training_args, extra_state_dict_dict):
    # first load loca model and then load global model
    with torch.no_grad():
        if 'zero3' in training_args.deepspeed:
            load_deepspeed(local_state_dict_list[client_id], model, strict=False)
        else:
            model.load_state_dict(local_state_dict_list[client_id], strict=False)
            
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
            
            weights[client_id] = -1e9
            weights = (weights/0.2).softmax(dim=0)
            
            sim_sum = weights.sum() - weights[client_id]
            
            # # weights[client_id] = sim_sum
            # # sim_sum += sim_sum
            
            for name in global_state_dict.keys():
                new_param = 0
                if 'lora1' in name:
                    target_key = name.replace('lora1', 'lora2')
                elif 'ia3_l_1' in name:
                    target_key = name.replace('ia3_l_1', 'ia3_l_2')
                
                for id in range(training_args.num_clients):
                    if id == client_id:
                        continue
                    else:
                        splited = target_key.split('.')
                        if int(splited[5]) != 0: # first layer
                            # if layer number is different
                            layer_num = []
                            for k in local_state_dict_list[id].keys():
                                if 'layers.' in k:
                                    layer_num.append(int(k.split('.')[5]))
                            layer_num = sorted(list(set(layer_num)))
                            
                            if int(splited[5]) != layer_num[-1]: # last layer
                                splited[5] = str(layer_num[-1])
                                target_key = '.'.join(splited)
                        new_param += weights[id]*local_state_dict_list[id][target_key] / sim_sum
                    
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

def feddualFMLpq_load_state_dict(model, global_state_dict, local_state_dict_list, client_id, training_args, extra_state_dict_dict):
    # first load loca model and then load global model
    with torch.no_grad():
        if 'zero3' in training_args.deepspeed:
            load_deepspeed(local_state_dict_list[client_id], model, strict=False)
        else:
            model.load_state_dict(local_state_dict_list[client_id], strict=False)
            
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
            
            weights[client_id] = -1e9
            weights = (weights/0.2).softmax(dim=0)
            
            sim_sum = weights.sum() - weights[client_id]
            
            # # weights[client_id] = sim_sum
            # # sim_sum += sim_sum
            cur_layer_num = []
            for k in global_state_dict.keys():
                if 'layers.' in k:
                    cur_layer_num.append(int(k.split('.')[5]))
            cur_layer_num = sorted(list(set(cur_layer_num)))
            
            for name in global_state_dict.keys():
                new_param = 0
                if 'lora1' in name:
                    target_key = name.replace('lora1', 'lora2')
                elif 'ia3_l_1' in name:
                    target_key = name.replace('ia3_l_1', 'ia3_l_2')
                
                for id in range(training_args.num_clients):
                    if id == client_id:
                        continue
                    else:
                        splited = target_key.split('.')
                        if int(splited[5]) != 0: # first layer
                            # if layer number is different
                            layer_num = []
                            for k in local_state_dict_list[id].keys():
                                if 'layers.' in k:
                                    layer_num.append(int(k.split('.')[5]))
                            layer_num = sorted(list(set(layer_num)))
                            
                            mid_layer = layer_num[int(len(layer_num)/2)] - 1
                            if cur_layer_num[-1] != layer_num[-1]: # if different size
                                if int(splited[5]) == cur_layer_num[-2]: # mid layer
                                    splited[5] = str(mid_layer)
                                elif int(splited[5]) == cur_layer_num[-1]: # last layer 
                                    splited[5] = str(layer_num[-1])
                                new_target_key = '.'.join(splited)
                            else:
                                new_target_key = target_key
                        else: 
                            new_target_key = target_key
                        new_param += weights[id]*local_state_dict_list[id][new_target_key] / sim_sum
                    
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

def feddualMultipq_load_state_dict(model, global_state_dict, local_state_dict_list, client_id, training_args, extra_state_dict_dict):
    # first load loca model and then load global model
    with torch.no_grad():
        if 'zero3' in training_args.deepspeed:
            load_deepspeed(local_state_dict_list[client_id], model, strict=False)
        else:
            model.load_state_dict(local_state_dict_list[client_id], strict=False)
            
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
            
            weights[client_id] = -1e9
            weights = (weights/0.2).softmax(dim=0)
            
            sim_sum = weights.sum() - weights[client_id]
            
            # # weights[client_id] = sim_sum
            # # sim_sum += sim_sum
            cur_layer_num = []
            for k in global_state_dict.keys():
                if 'layers.' in k:
                    cur_layer_num.append(int(k.split('.')[5]))
            cur_layer_num = sorted(list(set(cur_layer_num)))
            
            for name in global_state_dict.keys():
                new_param = 0
                if 'lora1' in name:
                    target_key = name.replace('lora1', 'lora2')
                elif 'ia3_l_1' in name:
                    target_key = name.replace('ia3_l_1', 'ia3_l_2')
                
                for id in range(training_args.num_clients):
                    if id == client_id:
                        continue
                    else:
                        splited = target_key.split('.')
                        # if layer number is different
                        layer_num = []
                        for k in local_state_dict_list[id].keys():
                            if 'layers.' in k:
                                layer_num.append(int(k.split('.')[5]))
                        layer_num = len(set(layer_num)) // 4
                        
                        target_layers = [layer_num*1 -1,layer_num*2 -1,layer_num*3 -1,layer_num*4 -1]
                        if cur_layer_num[-1] != target_layers[-1]: # if different size
                            if int(splited[5]) == cur_layer_num[0]: # mid layer
                                splited[5] = str(target_layers[0])
                            elif int(splited[5]) == cur_layer_num[1]: # last layer 
                                splited[5] = str(target_layers[1])
                            elif int(splited[5]) == cur_layer_num[2]: # last layer 
                                splited[5] = str(target_layers[2])
                            elif int(splited[5]) == cur_layer_num[3]: # last layer 
                                splited[5] = str(target_layers[3])
                            new_target_key = '.'.join(splited)
                        else:
                            new_target_key = target_key
                        new_param += weights[id]*local_state_dict_list[id][new_target_key] / sim_sum
                    
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

def feddualMulti2pq_load_state_dict(model, global_state_dict, local_state_dict_list, client_id, training_args, extra_state_dict_dict):
    # first load loca model and then load global model
    with torch.no_grad():
        if 'zero3' in training_args.deepspeed:
            load_deepspeed(local_state_dict_list[client_id], model, strict=False)
        else:
            model.load_state_dict(local_state_dict_list[client_id], strict=False)
            
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
            
            weights[client_id] = -1e9
            weights = (weights/0.2).softmax(dim=0)
            
            sim_sum = weights.sum() - weights[client_id]
            
            # # weights[client_id] = sim_sum
            # # sim_sum += sim_sum
            cur_layer_num = []
            for k in global_state_dict.keys():
                if 'layers.' in k:
                    cur_layer_num.append(int(k.split('.')[5]))
            cur_layer_num = sorted(list(set(cur_layer_num)))
            
            for name in global_state_dict.keys():
                new_param = 0
                if 'lora1' in name:
                    target_key = name.replace('lora1', 'lora2')
                elif 'ia3_l_1' in name:
                    target_key = name.replace('ia3_l_1', 'ia3_l_2')
                
                for id in range(training_args.num_clients):
                    if id == client_id:
                        continue
                    else:
                        splited = target_key.split('.')
                        # if layer number is different
                        layer_num = []
                        for k in local_state_dict_list[id].keys():
                            if 'layers.' in k:
                                layer_num.append(int(k.split('.')[5]))
                        layer_num = len(set(layer_num)) // 4
                        
                        target_layers = [layer_num*1 -2, layer_num*1 -1, layer_num*2 -2, layer_num*2 -1,
                                         layer_num*3 -2, layer_num*3 -1, layer_num*4 -2, layer_num*4 -1]
                        if cur_layer_num[-1] != target_layers[-1]: # if different size
                            if int(splited[5]) == cur_layer_num[0]: # mid layer
                                splited[5] = str(target_layers[0])
                            elif int(splited[5]) == cur_layer_num[1]: # last layer 
                                splited[5] = str(target_layers[1])
                            elif int(splited[5]) == cur_layer_num[2]: # last layer 
                                splited[5] = str(target_layers[2])
                            elif int(splited[5]) == cur_layer_num[3]: # last layer 
                                splited[5] = str(target_layers[3])
                            elif int(splited[5]) == cur_layer_num[4]: # last layer 
                                splited[5] = str(target_layers[4])
                            elif int(splited[5]) == cur_layer_num[5]: # last layer 
                                splited[5] = str(target_layers[5])
                            elif int(splited[5]) == cur_layer_num[6]: # last layer 
                                splited[5] = str(target_layers[6])
                            elif int(splited[5]) == cur_layer_num[7]: # last layer 
                                splited[5] = str(target_layers[7])
                            new_target_key = '.'.join(splited)
                        else:
                            new_target_key = target_key
                        new_param += weights[id]*local_state_dict_list[id][new_target_key] / sim_sum
                    
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
