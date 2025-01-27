from typing import Callable, Tuple, Type, Dict

from federated_methods.fedavg import fedavg_load_state_dict, fedavg_aggregate_state_dict, fedavg_create_trainer, fedavg_memefficient_aggregate_state_dict, fedavg_memefficient_load_state_dict
from federated_methods.sft import sft_load_state_dict
from federated_methods.task_id import task_id_create_trainer
from federated_methods.fedours import fedours_ema_distill_create_trainer, fedours_load_state_dict, OURS_set_state_dict, OURS_aggregate_state_dict, OURS_memefficient_aggregate_state_dict, fedours_memefficient_load_state_dict

from federated_methods.fedpq import (feddualpq_load_state_dict, fedpq_load_state_dict, fedduallastpq_load_state_dict, 
                                     fedlastpq_load_state_dict, fedFLpq_load_state_dict, feddualFLpq_load_state_dict, 
                                     fedFMLpq_load_state_dict, feddualFMLpq_load_state_dict, feddualMultipq_load_state_dict,
                                     feddualMultipq2_load_state_dict, fedMultipq_load_state_dict,
                                     fedlastpq_tv_load_state_dict, fedMultipq_tv_load_state_dict)

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
    elif mode =='fedpq' or mode == 'fedpqfreeze' or mode == 'fedpqfreeze2':
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, fedpq_load_state_dict, fedavg_create_trainer, fedavg_aggregate_state_dict
    elif mode =='fedpq_sft' or mode == 'fedpqfreeze_sft' or mode == 'fedpqfreeze2_sft' or mode =='fedlastpqfullfreeze_sft' or mode == 'fedMultipqfullfreeze_sft':
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, sft_load_state_dict, fedavg_create_trainer, fedavg_aggregate_state_dict
    elif mode == 'fedlastpq' or mode == 'fedlastpqfreeze' or mode =='fedlastpqfullfreeze':
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, fedlastpq_load_state_dict, fedavg_create_trainer, fedavg_aggregate_state_dict
    elif mode =='fedlastpqfullfreeze_tv' or mode == 'fedlastpqfullfreeze_ours':
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, fedlastpq_tv_load_state_dict, fedavg_create_trainer, fedavg_aggregate_state_dict
    elif mode == 'fedFLpq' or mode == 'fedFLpqfreeze':
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, fedFLpq_load_state_dict, fedavg_create_trainer, fedavg_aggregate_state_dict
    elif mode == 'fedFMLpq':
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, fedFMLpq_load_state_dict, fedavg_create_trainer, fedavg_aggregate_state_dict
    elif mode =='feddualpq' or mode == 'feddualpqfullfreeze' or mode == 'feddualpqfullfreeze_tv':
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, feddualpq_load_state_dict, fedours_ema_distill_create_trainer, OURS_aggregate_state_dict
    elif mode =='fedduallastpq' or mode == 'fedduallastpqfreeze' or mode == 'fedduallastpqfullfreeze' or mode == 'fedduallastpqfullfreeze_tv':
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, fedduallastpq_load_state_dict, fedours_ema_distill_create_trainer, OURS_aggregate_state_dict
    elif mode =='feddualFLpq' or mode == 'feddualFLpqfreeze':
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, feddualFLpq_load_state_dict, fedours_ema_distill_create_trainer, OURS_aggregate_state_dict
    elif mode =='feddualFMLpq':
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, feddualFMLpq_load_state_dict, fedours_ema_distill_create_trainer, OURS_aggregate_state_dict
    elif mode =='feddualMultipqfreeze' or mode =='feddualMultipqfullfreeze' or mode == 'feddualMultipqfullfreeze_tv':
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, feddualMultipq_load_state_dict, fedours_ema_distill_create_trainer, OURS_aggregate_state_dict
    elif mode =='feddualMultipqfreeze2':
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, feddualMultipq2_load_state_dict, fedours_ema_distill_create_trainer, OURS_aggregate_state_dict
    
    elif mode =='fedMultipqfullfreeze':
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, fedMultipq_load_state_dict, fedavg_create_trainer, OURS_aggregate_state_dict
    elif mode =='fedMultipqfullfreeze_tv' or 'fedMultipqfullfreeze_ours':
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, fedMultipq_tv_load_state_dict, fedavg_create_trainer, OURS_aggregate_state_dict
    
    
    elif mode =='fedavg_memefficient':
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, fedavg_memefficient_load_state_dict, fedavg_create_trainer, fedavg_memefficient_aggregate_state_dict
    elif mode == 'fedours_memefficient':
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, fedours_memefficient_load_state_dict, fedours_ema_distill_create_trainer, OURS_memefficient_aggregate_state_dict 
    else:
        raise NotImplementedError(mode)
    return set_state_dict, load_state_dict, create_trainer, aggregate_state_dict, extra_modules
