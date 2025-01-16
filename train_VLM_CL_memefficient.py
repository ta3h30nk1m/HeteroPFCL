import logging.config
import os
import random
import gc
import shutil

import numpy as np
import torch
from configuration.VLM_config_new import ModelArguments, DataArguments, TrainingConfig
import transformers
from utils.train_utils import get_VLMmodel, get_peft_state_maybe_zero_3, get_peft_state_non_lora_maybe_zero_3, get_task_vectors, load_deepspeed, configure_online_datastream

from federated_methods.method_manager import select_method
from utils.data_loader_VLM import LazySupervisedDataset, DataCollatorForSupervisedDataset
from typing import Dict

import copy
import json
from transformers import BitsAndBytesConfig
import time
import datetime
import torch.nn.functional as F

from models.coda_prompt import CodaPrompt

def main():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingConfig))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))    
    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_skip_modules=["mm_projector"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type # {'fp4', 'nf4'}
            )
        ))

    logging.config.fileConfig("./configuration/logging.conf")
    logger = logging.getLogger()

    os.makedirs(f"results/{training_args.mode}/{training_args.note}", exist_ok=True)
    fileHandler = logging.FileHandler(f'results/{training_args.mode}/{training_args.note}/seed_{training_args.seed}.log', mode="w")

    formatter = logging.Formatter(
        "[%(levelname)s] %(filename)s:%(lineno)d > %(message)s"
    )
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)
    if training_args.local_rank == 0 or training_args.local_rank == -1: 
        logger.info(training_args)

    # Fix the random seeds
    torch.manual_seed(training_args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(training_args.seed)
    random.seed(training_args.seed)
    
    
    train_datalists, test_datalists = get_datalists(training_args, training_args.scenario)
    
    # select functions
    set_state_dict, load_state_dict, create_trainer, aggregate_state_dict, extra_modules = select_method(training_args.mode)
    
    # create folder
    training_args.state_dir = training_args.state_dir + '_' + training_args.note
    if not os.path.exists(training_args.state_dir):
        os.makedirs(training_args.state_dir)
    
    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {'use_reentrant':False}
    
    
    model_ids = {}
    model_list = {}
    models = {}
    global_state_dict_list = []
    local_state_dict_list = []
    old_local_state_dict_list = []
    os.makedirs(training_args.state_dir + '/global_state_dict', exist_ok=True)
    os.makedirs(training_args.state_dir + '/local_state_dict', exist_ok=True)
    os.makedirs(training_args.state_dir + '/old_local_state_dict', exist_ok=True)
    for client_id in range(len(train_datalists)):
        train_datalist = train_datalists[client_id]
        model_id = train_datalist[0]['model_id']
        
        global_state_dict_filename = training_args.state_dir + f'/global_state_dict/client_{client_id}.pth'
        local_state_dict_filename = training_args.state_dir + f'/local_state_dict/client_{client_id}.pth'
        old_local_state_dict_filename = training_args.state_dir + f'/old_local_state_dict/client_{client_id}.pth'

        global_state_dict_list.append(global_state_dict_filename)
        local_state_dict_list.append(local_state_dict_filename)
        old_local_state_dict_list.append(old_local_state_dict_filename)
        if model_id in model_list.keys():
            torch.save(model_list[model_id], local_state_dict_filename)
            torch.save(model_list[model_id], old_local_state_dict_filename)
            global_state_dict = copy.deepcopy(model_list[model_id])
            if training_args.mode == 'fedours':
                keys_to_del = []
                for k in global_state_dict.keys():
                    if 'lora2' in k or 'ia3_l_2' in k or 'ia3_generator_2' in k or 'lang_prompt_ia3_pool_2' in k \
                    or 'lang_prompt_dap_key_embeddings_2' in k or 'lang_prompt_downsample_2' in k or 'lang_prompt_norm_2' in k \
                    or 'lang_prompt_downsample_kv_2' in k or 'lang_prompt_downsample_mlp_2' in k\
                    or 'w_gate' in k or 'w_noise' in k:
                        keys_to_del.append(k)
                for k in keys_to_del:
                    del global_state_dict[k]
            torch.save(global_state_dict, global_state_dict_filename)
            
            model_ids[model_id].append(client_id)
        else:
            new_model_args = copy.deepcopy(model_args)
            new_model_args.model_name_or_path = model_id
            model, tokenizer, processor, new_data_args = get_VLMmodel(new_model_args, training_args, bnb_model_from_pretrained_args, data_args)
            
            if training_args.load_checkpoint is not None and not training_args.fedours:
                logger.info(f'load {training_args.load_checkpoint}')
                server_state_dict = torch.load(training_args.load_checkpoint, map_location='cpu')
                
                with torch.no_grad():
                    model.load_state_dict(server_state_dict, strict=False)
                
                if ('fedours' in training_args.load_checkpoint) and training_args.mode not in ['fedours', 'ours_generator', 'ours_generator2']:
                    local_state_dict = {}
                    for name in server_state_dict.keys():
                        if 'lora1' in name:
                            target_key = name.replace('lora1', 'lora')
                        elif 'ia3_l_1' in name:
                            target_key = name.replace('ia3_l_1', 'ia3_l')
                        local_state_dict[target_key] = server_state_dict[name]
                    
                    server_state_dict = local_state_dict
                
                with torch.no_grad():
                    model.load_state_dict(server_state_dict, strict=False)
                    
                if training_args.mode in ['fedours', 'ours_generator', 'ours_generator2']:
                    local_state_dict = {}
                    for name in server_state_dict.keys():
                        if 'lora1' in name:
                            target_key = name.replace('lora1', 'lora2')
                        elif 'ia3_l_1' in name:
                            target_key = name.replace('ia3_l_1', 'ia3_l_2')
                        local_state_dict[target_key] = server_state_dict[name]
                    
                    model.load_state_dict(local_state_dict, strict=False)
            
            
            global_state_dict = get_peft_state_maybe_zero_3(
                        model.named_parameters(), training_args.lora_bias
                    )
            non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
                model.named_parameters()
            )
            global_state_dict.update(non_lora_state_dict)
            
            torch.save(global_state_dict, local_state_dict_filename)
            torch.save(global_state_dict, old_local_state_dict_filename)
            new_global_state_dict=copy.deepcopy(global_state_dict)
            if training_args.mode == 'fedours':
                keys_to_del = []
                for k in new_global_state_dict.keys():
                    if 'lora2' in k or 'ia3_l_2' in k or 'ia3_generator_2' in k or 'lang_prompt_ia3_pool_2' in k \
                    or 'lang_prompt_dap_key_embeddings_2' in k or 'lang_prompt_downsample_2' in k or 'lang_prompt_norm_2' in k \
                    or 'lang_prompt_downsample_kv_2' in k or 'lang_prompt_downsample_mlp_2' in k\
                    or 'w_gate' in k or 'w_noise' in k:
                        keys_to_del.append(k)
                for k in keys_to_del:
                    del new_global_state_dict[k]
            torch.save(new_global_state_dict, global_state_dict_filename)
            
            model_list[model_id] = global_state_dict
            
            models[model_id] = model
            
            model_ids[model_id] = [client_id]
    
    del model_list, global_state_dict, new_global_state_dict
    extra_state_dict_dict = {}
    
    if training_args.fedours:
        logger.info(f'load task vector {training_args.load_checkpoint}')
        tv_weights = torch.load(training_args.load_checkpoint, map_location='cpu')
        prev_task_vectors = tv_weights['task_vectors']
        prev_local_state_dict_list = tv_weights['local_state_dict_list']
        
        current_task_vectors = get_task_vectors(model, tokenizer, processor, train_datalists, training_args, data_args, global_state_dict_list, make_supervised_data_module)
    else:
        current_task_vectors = None

    training_loss = [[] for i in range(training_args.num_clients)]
    
    # start federated learning
    start_time = time.time()
    frac_clients = 1
    
    memory = [[] for id in range(training_args.num_clients)]
    memory_count = [np.array([]) for id in range(training_args.num_clients)]
    memory_size = training_args.memory_size
    total_batchsize = training_args.per_gpu_train_batch_size*training_args.world_size*training_args.gradient_accumulation_steps
    init_lr = training_args.learning_rate
    mm_init_lr = training_args.mm_projector_lr
    final_lr = training_args.final_lr
    mm_final_lr = training_args.mm_final_lr
    
    total_rounds = training_args.num_rounds * training_args.num_tasks
    last_task_id = [-1 for _ in range(training_args.num_clients)]
    fisher_olds = [None for _ in range(training_args.num_clients)]
    task_vectors = [None for _ in range(training_args.num_clients)]
    
    lr_step = (init_lr - final_lr)/total_rounds
    mm_lr_step = (mm_init_lr - mm_final_lr)/total_rounds
    for curr_round in range(total_rounds):
        # copy current local state dicts to old
        for i in range(len(local_state_dict_list)):
            shutil.copyfile(local_state_dict_list[i], old_local_state_dict_list[i]) 


        if curr_round > 0 and training_args.use_task_vector:
            path = os.path.join(training_args.state_dir, f"round{curr_round}_task_vector_local_weights.pth")
            tv_weight = {'task_vectors': task_vectors, 'local_state_dict_list': old_local_state_dict_list}
            torch.save(tv_weight, path)
            
            # NEW: SVD to match the gradient size
            if training_args.is_hetero_model:
                breakpoint()
                U, S, Vh = torch.linalg.svd(task_vectors, full_matrices=False)
            
            
            # vectorize cosine sim and then average them
            sims = []
            for grad_idx in range(task_vectors[0].shape[-1]):
                task_vector = F.normalize(torch.stack([tv[:,grad_idx] for tv in task_vectors], dim=0), dim=-1)
                sim = torch.matmul(task_vector,
                                torch.transpose(task_vector, 1, 0))
                sim = torch.transpose(sim, 1, 0)
                sims.append(sim)
            
            sim = torch.stack(sims, dim=0).mean(dim=0)
            
            
            extra_state_dict_dict['task_similarity'] = sim
            print("task similarity matrix:")
            print(sim)
        
        # clients turn
        cids = np.arange(training_args.num_clients).tolist()
        num_selection = int(round(training_args.num_clients*frac_clients)) 
        selected_ids = sorted(random.sample(cids, num_selection)) 
        if training_args.local_rank == 0 or training_args.local_rank == -1: 
            logger.info(f"Round {curr_round} | selected_ids: {selected_ids}\n")
        
        # selected_ids = cids
        training_args.learning_rate = init_lr - lr_step*curr_round
        training_args.mm_projector_lr = mm_init_lr - mm_lr_step*curr_round
        if curr_round > 0 and training_args.is_wsd:
            training_args.warmup_ratio = 0
            training_args.warmup_steps = 0
        for idx in range(num_selection):
            client_id = selected_ids[idx]
            
            ##### simulate online memory insertion & get_batch ####
            sub_dataset = train_datalists[client_id][curr_round]['datalist']
            num_iterations = train_datalists[client_id][curr_round]['num_iter']
            
            task_id = train_datalists[client_id][curr_round]['task_id']
            
            test_datalist = test_datalists[client_id]
            
            model_id = train_datalists[client_id][curr_round]['model_id']
            new_model_args = copy.deepcopy(model_args)
            new_model_args.model_name_or_path = model_id
            new_data_args = copy.deepcopy(data_args)
            new_data_args.model_name_for_dataarg = model_id
            # model,_,_,_ = get_VLMmodel(new_model_args, training_args, bnb_model_from_pretrained_args, new_data_args)
            model = models[model_id]
            
            extra_state_dict_dict['client_id'] = client_id
            extra_state_dict_dict['curr_round'] = curr_round
            extra_state_dict_dict['test_datalist'] = test_datalist
            extra_state_dict_dict['processor'] = processor
            extra_state_dict_dict['data_args'] = copy.deepcopy(new_data_args)
            if training_args.use_task_id:
                extra_state_dict_dict['task_id'] = task_id
            
            load_state_dict(model, global_state_dict_list[client_id], old_local_state_dict_list, client_id, training_args, extra_state_dict_dict)
            print('model loading done')
            
            if training_args.fedours:
                sims = []
                for grad_idx in range(prev_task_vectors[0].shape[-1]):
                    task_vector = F.normalize(torch.stack([tv[:,grad_idx] for tv in prev_task_vectors] + [current_task_vectors[client_id][:,grad_idx]], dim=0), dim=-1)
                    sim = torch.matmul(task_vector,
                                    torch.transpose(task_vector, 1, 0))
                    sim = torch.transpose(sim, 1, 0)
                    sims.append(sim)
                
                sim = torch.stack(sims, dim=0).mean(dim=0)
                
                print(sim)
                new_global_state_dict = {}
            
                weights = sim[-1][:-1].clone()
                
                weights = (weights/0.2).softmax(dim=0)
                
                sim_sum = weights.sum()
                
                for name in global_state_dict.keys():
                    new_param = 0
                    if training_args.mode in ['fedours', 'ours_generator', 'ours_generator2']:
                        if 'lora1' in name:
                            target_key = name.replace('lora1', 'lora2')
                        elif 'ia3_l_1' in name:
                            target_key = name.replace('ia3_l_1', 'ia3_l_2')
                    else:
                        if 'lora' in name:
                            target_key = name.replace('lora', 'lora2')
                        elif 'ia3_l' in name:
                            target_key = name.replace('ia3_l', 'ia3_l_2')
                    for id in range(len(prev_local_state_dict_list)):
                        new_param += weights[id]*prev_local_state_dict_list[id][target_key] / sim_sum
                    
                    new_global_state_dict[name] = new_param
                    if training_args.mode in ['fedours', 'ours_generator', 'ours_generator2']: 
                        new_global_state_dict[target_key] = new_param
                
                if 'zero3' in training_args.deepspeed:
                    load_deepspeed(new_global_state_dict, model, strict=False)
                else:
                    model.load_state_dict(new_global_state_dict, strict=False) 
            
            datalist = configure_online_datastream(sub_dataset, num_iterations, training_args, client_id, memory, memory_count, memory_size, total_batchsize)           
            data_module = make_supervised_data_module(client_data=datalist, # sub_dataset
                                                tokenizer=tokenizer,
                                                processor=processor,
                                                data_args=copy.deepcopy(new_data_args))
            
            if training_args.local_rank == 0 or training_args.local_rank == -1: 
                logger.info(f'Round {curr_round} | train client {client_id} | num samples {len(sub_dataset)}')

            # ===== Train local model on the client side =====
            if training_args.use_fisher:
                extra_state_dict_dict['fisher_old'] = fisher_olds[client_id]
                
            if training_args.use_task_vector:
                extra_state_dict_dict['task_vector'] = task_vectors[client_id]
            
            trainer = create_trainer(model, tokenizer, training_args, data_module, extra_state_dict_dict)

            results = trainer.train()
            training_loss[client_id].append(results.training_loss)
            if training_args.use_fisher:
                fisher_olds[client_id] = trainer.fisher_old
            
            if training_args.use_task_vector:
                task_vectors[client_id] = trainer.task_vector #- original_weights
            
            if training_args.local_rank == 0 or training_args.local_rank == -1: 
                path = os.path.join(training_args.state_dir, f"{client_id}_trainer_state.json")
                trainer.state.save_to_json(path)
            
            model.config.use_cache = True
            
            # save local model
            output_dir = os.path.join(training_args.state_dir, f"{client_id}_client_model_round{curr_round+1}.pth")
            if training_args.lora_enable:
                state_dict = get_peft_state_maybe_zero_3(
                    model.named_parameters(), training_args.lora_bias
                )
                non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
                    model.named_parameters()
                )
                state_dict.update(non_lora_state_dict)
            else:
                state_dict = {k: t.detach().cpu().clone() for k, t in model.named_parameters() if t.requires_grad}
            
            # local_state_dict_list[client_id] = copy.deepcopy(state_dict)
            torch.save(state_dict, local_state_dict_list[client_id])
            
            if (training_args.local_rank == 0 or training_args.local_rank == -1):
                torch.save(state_dict, output_dir)
            
            trainer.deepspeed.empty_partition_cache()
            trainer.accelerator.free_memory()
            del trainer
            model = model.cpu()
            gc.collect()
            torch.cuda.empty_cache()
            logger.info(f"done Round {curr_round} client {client_id} | elapsed time {datetime.timedelta(seconds=int(time.time() - start_time))} | ")
            
        
        aggregate_state_dict(global_state_dict_list, local_state_dict_list, selected_ids, num_selection, training_args, **extra_state_dict_dict)
        
        # Save server model
        if training_args.mode != 'fedours':
            if (training_args.local_rank == 0 or training_args.local_rank == -1): 
                torch.save(global_state_dict_list[0], os.path.join(training_args.state_dir, f"server_model_round{curr_round}.pth"))
            
    if training_args.use_task_vector:
        path = os.path.join(training_args.state_dir, f"round{curr_round+1}_task_vector_local_weights.pth")
        tv_weight = {'task_vectors': task_vectors, 'local_state_dict_list': local_state_dict_list}
        torch.save(tv_weight, path)
        
        # task_vector = F.normalize(torch.stack(task_vectors, dim=0), dim=-1)
        # sim = torch.matmul(task_vector,
        #                 torch.transpose(task_vector, 1, 0))
        # sim = torch.transpose(sim, 1, 0)
        # sim = (sim+1)/2
        
        sims = []
        for grad_idx in range(task_vectors[0].shape[-1]):
            task_vector = F.normalize(torch.stack([tv[:,grad_idx] for tv in task_vectors], dim=0), dim=-1)
            sim = torch.matmul(task_vector,
                            torch.transpose(task_vector, 1, 0))
            sim = torch.transpose(sim, 1, 0)
            sims.append(sim)
        
        sim = torch.stack(sims, dim=0).mean(dim=0)
        
        extra_state_dict_dict['task_similarity'] = sim
        extra_state_dict_dict['curr_round'] += 1
        for client_id in range(training_args.num_clients):
            load_state_dict(model, global_state_dict, local_state_dict_list, client_id, training_args, extra_state_dict_dict)
    logger.info("total done\n")

def make_supervised_data_module(client_data, tokenizer: transformers.PreTrainedTokenizer, processor,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(client_data, tokenizer, data_args, processor)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)

def get_datalists(args, scenario_num):
    with open(f"./scenarios/scenario-{scenario_num}.json") as fp:
        scenario = json.load(fp)
    assert args.num_clients == len(scenario)

    train_datalists = {}
    test_datalists = {}
    
    max_iterations = args.num_iter
    rounds_per_task = args.num_rounds

    for client_data in scenario:
        client_id = client_data['client_id']
        train_datalist = []
        test_datalist = []
        for task_id, data in enumerate(client_data['datasets']):
            with open(f"./dataset/{data['dataset']}/train/dataset-{str(data['subset_id'])}.json") as fp:
                datalist = json.load(fp)
            random.shuffle(datalist)
            samplenum_per_rounds = int(len(datalist) / rounds_per_task)
            num_iter = max_iterations #max(int(max_iterations*samplenum_per_rounds/2000), 2) # 10000 / 5 = 2000
            for i in range(rounds_per_task):
                train_datalist.append(
                    {'datalist':datalist[i*samplenum_per_rounds:(i+1)*samplenum_per_rounds],
                     'num_iter': num_iter,
                     'task_id': task_id,
                     'model_id': client_data['model_id']})
            with open(f"./dataset/{data['dataset']}/test/dataset-{str(data['subset_id'])}.json") as fp:
                datalist = json.load(fp)
            test_datalist.append({
                "data_name": f"{data['dataset']}-{data['subset_id']}",
                "type": data['type'],
                "data": datalist,
                "train_start_round": rounds_per_task*task_id})
            
            train_datalists[client_id] = train_datalist
        test_datalists[client_id] = test_datalist

    return train_datalists, test_datalists

if __name__ == "__main__":
    main()