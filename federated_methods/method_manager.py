from typing import Callable, Tuple, Type, Dict

from federated_methods.fedavg import fedavg_load_state_dict, fedavg_aggregate_state_dict, fedavg_create_trainer, fedavg_memefficient_aggregate_state_dict, fedavg_memefficient_load_state_dict
from federated_methods.sft import sft_load_state_dict
from federated_methods.task_id import task_id_create_trainer
from federated_methods.fedours import fedours_ema_distill_create_trainer, fedours_load_state_dict, OURS_set_state_dict, OURS_aggregate_state_dict, OURS_memefficient_aggregate_state_dict, fedours_memefficient_load_state_dict

from federated_methods.fedpq import feddualpq_load_state_dict, fedpq_load_state_dict, fedduallastpq_load_state_dict, fedlastpq_load_state_dict

def dummy_function(*args):
    return {}

def select_method(mode: str) -> Tuple[Callable, Callable, Callable, Callable, Dict]:
    extra_modules = {}
    if mode == 'sft':
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, sft_load_state_dict, fedavg_create_trainer, fedavg_aggregate_state_dict
    elif mode == 'fedavg':
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, fedavg_load_state_dict, fedavg_create_trainer, fedavg_aggregate_state_dict
    elif mode == 'fedours':
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, fedours_load_state_dict, fedours_ema_distill_create_trainer, OURS_aggregate_state_dict  
    elif mode =='fedpq':
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, fedpq_load_state_dict, fedavg_create_trainer, fedavg_aggregate_state_dict
    elif mode == 'fedlastpq':
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, fedlastpq_load_state_dict, fedavg_create_trainer, fedavg_aggregate_state_dict
    elif mode =='feddualpq':
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, feddualpq_load_state_dict, fedours_ema_distill_create_trainer, OURS_aggregate_state_dict
    elif mode =='fedduallastpq':
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, fedduallastpq_load_state_dict, fedours_ema_distill_create_trainer, OURS_aggregate_state_dict
        
    elif mode =='fedavg_memefficient':
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, fedavg_memefficient_load_state_dict, fedavg_create_trainer, fedavg_memefficient_aggregate_state_dict
    elif mode == 'fedours_memefficient':
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, fedours_memefficient_load_state_dict, fedours_ema_distill_create_trainer, OURS_memefficient_aggregate_state_dict 
    else:
        raise NotImplementedError(mode)
    return set_state_dict, load_state_dict, create_trainer, aggregate_state_dict, extra_modules
