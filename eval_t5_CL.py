import logging.config
import os
import random
import re
import string

import numpy as np
import torch
from configuration.VLM_config_new import ModelArguments, DataArguments, TrainingConfig
import transformers
from utils.train_utils import get_VLMmodel
import copy

import json
from transformers import BitsAndBytesConfig

from utils.data_loader_VLM import GenerationDataset, DataCollatorForGenerationDataset
from torch.utils.data import DataLoader
from utils.eval_metrics import NLPEvaluator, matching_token_num
from tqdm import tqdm

from models.llava.mm_utils import KeywordsStoppingCriteria
from models.llava import conversation as conversation_lib_llava

import warnings
import time
import datetime
warnings.filterwarnings('ignore')
os.environ["TOKENIZERS_PARALLELISM"] = "true"

from transformers import Trainer
from utils.train_utils import preprocess_dataset, SeqToSeqEncode

ALPHABET = ['A','B','C','D','E','F']

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
        
    # Fix the random seeds
    torch.manual_seed(training_args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(training_args.seed)
    random.seed(training_args.seed)
    torch.cuda.manual_seed(training_args.seed)
    torch.cuda.manual_seed_all(training_args.seed)

    logging.config.fileConfig("./configuration/logging.conf")
    logger = logging.getLogger()

    os.makedirs(f"eval_results/{training_args.mode}/{training_args.note}", exist_ok=True)
    fileHandler = logging.FileHandler(f'eval_results/{training_args.mode}/{training_args.note}/round_{training_args.round_to_eval}.log', mode="w")

    # writer = SummaryWriter(f'tensorboard/{training_args.mode}/{training_args.note}/federated')

    formatter = logging.Formatter(
        "[%(levelname)s] %(filename)s:%(lineno)d > %(message)s"
    )
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)
    logger.info(training_args)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print(device)
    logger.info(f"Set the device ({device})")

    # model, tokenizer, processor, data_args = get_VLMmodel(model_args, training_args, bnb_model_from_pretrained_args, data_args)
    
    _, test_datalists = get_datalists(training_args, training_args.scenario)
    
    batch_size = 4 #if 'l2p' in training_args.mode or 'dap' in training_args.mode or 'LAE' in training_args.mode or 'fedsim' in training_args.mode else 4
    
    logger.info(f'Evaluatiing clients and server at round {training_args.round_to_eval}')
    start_time = time.time()
    server_eval_key = []
    
    if not training_args.zeroshot and training_args.eval_server:
        logger.info(f'load ./client_states_{training_args.note}/server_model_round{training_args.round_to_eval-1}.pth')
        server_state_dict = torch.load(f'./client_states_{training_args.note}/server_model_round{training_args.round_to_eval-1}.pth', map_location='cpu')
    
    for client_id in range(training_args.num_clients):
        if training_args.eval_client is not None:
            if client_id != training_args.eval_client:
                continue
        if training_args.eval_client_start is not None and training_args.eval_client_end is not None:
            if client_id < training_args.eval_client_start or client_id > training_args.eval_client_end:
                continue
        # load client weight
        if not training_args.zeroshot:
            try:
                if training_args.eval_iter is not None:
                    logger.info(f'load ./client_states_{training_args.note}/{client_id}_client_model_round{training_args.round_to_eval}_itr{training_args.eval_iter}.pth')
                    client_state_dict = torch.load(f'./client_states_{training_args.note}/{client_id}_client_model_round{training_args.round_to_eval}_itr{training_args.eval_iter}.pth', map_location='cpu')    
                else:
                    logger.info(f'load ./client_states_{training_args.note}/{client_id}_client_model_round{training_args.round_to_eval}.pth')
                    client_state_dict = torch.load(f'./client_states_{training_args.note}/{client_id}_client_model_round{training_args.round_to_eval}.pth', map_location='cpu')
            except Exception as e:
                print(e)
                continue
        
        test_datalist = test_datalists[client_id]
        new_model_args = copy.deepcopy(model_args)
        new_model_args.model_name_or_path = test_datalist[0]['model_id']
        new_data_args = copy.deepcopy(data_args)
        new_data_args.model_name_for_dataarg = test_datalist[0]['model_id']
        model, tokenizer, processor, data_args = get_VLMmodel(new_model_args, training_args, bnb_model_from_pretrained_args, new_data_args)
        print(f"Model name: {test_datalist[0]['model_id']}")
        if training_args.eval_all:
            if not training_args.zeroshot:
                model.load_state_dict(client_state_dict, strict=False)
                model = model.to(torch.bfloat16)
                model = model.to(device)
            # if 'fedours' in training_args.mode or 'dual' in training_args.mode or 'fedquad' in training_args.mode or 'fedhexa' in training_args.mode or 'feddat' in training_args.mode or 'perada' in training_args.mode or 'ditto' in training_args.mode or 'feddpa' in training_args.mode:
            if 'perada' in training_args.mode or 'ditto' in training_args.mode:
                model.set_state('lora2')
            elif 'feddpa' in training_args.mode or 'feddat' in training_args.mode:
                model.set_state('gate')
            elif 'fedours' in training_args.mode or 'dual' in training_args.mode or 'fedquad' in training_args.mode or 'fedhexa' in training_args.mode or 'feddat' in training_args.mode:
                model.set_state(training_args.set_state)
            
            for client_id_ in range(training_args.num_clients):
                if training_args.eval_client_eval_start is not None and training_args.eval_client_eval_end is not None:
                    if client_id_ < training_args.eval_client_eval_start or client_id_ > training_args.eval_client_eval_end:
                        continue
                test_datalist_ = test_datalists[client_id_]
                for data_info in test_datalist_:
                    dataset = preprocess_dataset(data_info['data'])
                    dataset.set_transform(lambda x: SeqToSeqEncode(x, tokenizer, 128))
                    if (training_args.eval_iter is not None and os.path.isfile(f"./eval_results/{training_args.mode}/{training_args.note}/client{client_id}_round{training_args.round_to_eval}_iter{training_args.eval_iter}_{data_info['data_name']}-{client_id_}.json")) \
                        or (training_args.eval_iter is None and os.path.isfile(f"./eval_results/{training_args.mode}/{training_args.note}/client{client_id}_round{training_args.round_to_eval}_{data_info['data_name']}-{client_id_}.json")):
                        print('output file already exist')
                        continue
                    
                    trainer = Trainer(
                        model=model,
                        args=training_args,
                        train_dataset=dataset,
                        eval_dataset=dataset,
                        compute_metrics=compute_metrics_soft if data_info['type'] in ['cola', 'mrpc'] else compute_metrics_hard,
                    )
                    
                    if 'fedsim' in training_args.mode:
                        from federated_methods.fedsim import LLaVATrainerFEDSIM
                        import types
                        trainer.compute_loss = types.MethodType(LLaVATrainerFEDSIM.compute_loss, trainer)
                    
                    initial_metrics = trainer.evaluate()                 # runs on valid_dataset
                    if training_args.eval_iter is not None:
                        outputpath = f"./eval_results/{training_args.mode}/{training_args.note}/client{client_id}_round{training_args.round_to_eval}_iter{training_args.eval_iter}_{data_info['data_name']}-{client_id_}.json"
                    elif training_args.eval_iter is None:
                        outputpath = f"./eval_results/{training_args.mode}/{training_args.note}/client{client_id}_round{training_args.round_to_eval}_{data_info['data_name']}-{client_id_}.json"
                    with open(outputpath, 'w') as f:
                        json.dump(initial_metrics, f)
                    
            continue
        del model
    
    logger.info(f"elapsed time {datetime.timedelta(seconds=int(time.time() - start_time))} | ")

from transformers.trainer_utils import PredictionOutput
def compute_metrics_hard(p: PredictionOutput): # 
    predictions = p.predictions
    label_ids = p.label_ids # shape (batch_size, seq_len)
    # Hard metric: the model must output exactly the same as the target
    # This should be the default evaluation metric for most tasks
    pred = np.argmax(predictions[0], axis=-1)
    num_correct = sum([np.array_equal(pred[i], label_ids[i]) for i in range(len(pred))])
    accuracy = num_correct / len(pred)

    return {"accuracy": accuracy}

def compute_metrics_soft(p: PredictionOutput): # for cola and mrpc
    predictions = p.predictions
    label_ids = p.label_ids # shape (batch_size, seq_len)
    # Soft metric: we limit the output space to the target space
    # i.e. the model classify the one with higher prob in positive and negative
    # **Use it in cola and mrpc, because it's too hard for vanilla lora**
    # Only suit for the binary classification with each label of 1 token
    label_ids = label_ids[:, 0] # remove the eos token
    unique_labels = np.unique(label_ids)
    flipped_labels = np.ones_like(label_ids) * unique_labels.sum() - label_ids
    predictions = predictions[0][:, 0, :] # remove the eos token # seq_len, tokens
    label_prob = predictions[np.arange(len(predictions)), label_ids]
    flipped_label_prob = predictions[np.arange(len(predictions)), flipped_labels]
    num_correct = sum(label_prob > flipped_label_prob)
    accuracy = num_correct / len(label_prob)
    
    return {"accuracy": accuracy}


from nlp_data import DATASET_MAP
def get_datalists(args, scenario_num):
    scenario = [
            {"client_id": 0,
             "datasets":["mnli", "qqp"],
             "model_id":"t5-base"},
            {"client_id": 1,
            "datasets":["mrpc", "rte"],
            "model_id":"t5-small"},
            {"client_id": 2,
            "datasets":["sst2", "qnli"],
            "model_id":"t5-base"},
            {"client_id": 3,
            "datasets":["cola", "mrpc"],
            "model_id":"t5-small"}
        ]
    
    if args.is_incremental_client_scenario:
        incremental_setup = scenario[0]
        assert args.num_rounds == incremental_setup['num_rounds']
        assert args.num_tasks == incremental_setup['num_tasks']
        assert args.num_rounds * args.num_tasks == len(incremental_setup['num_active_clients'])
        
        scenario = scenario[1:]
    else:
        incremental_setup = {
            "num_active_clients": [args.num_clients,]*int(args.num_rounds * args.num_tasks)
        }
    assert args.num_clients == len(scenario)

    train_datalists = {}
    test_datalists = {}
    
    max_iterations = args.num_iter
    rounds_per_task = args.num_rounds

    for client_data in scenario:
        client_id = client_data['client_id']
        train_datalist = []
        test_datalist = []
        eval_cnt = 0
        train_cnt = 0
        for data in client_data['datasets']:
            
            dataset_function = DATASET_MAP[data]
            trainset, valset, _ = dataset_function()
            datalist = []
            for item in valset:
                datalist.append(item)
            # random.shuffle(datalist)
            samplenum_per_rounds = int(len(datalist) / rounds_per_task)
            for i in range(int(rounds_per_task)):
                train_datalist.append(
                    {'datalist':[],
                     'train_cnt': train_cnt})
                train_cnt += samplenum_per_rounds
            print(len(datalist))
            test_datalist.append({
                "data_name": f"{data}",
                "type": f"{data}",
                "model_id": client_data['model_id'],
                "data": datalist[:10000],
                "eval_cnt": eval_cnt})
            eval_cnt += len(valset)
            
            train_datalists[client_id] = train_datalist
        test_datalists[client_id] = test_datalist
    
    return train_datalists, test_datalists

if __name__ == "__main__":
    main()
