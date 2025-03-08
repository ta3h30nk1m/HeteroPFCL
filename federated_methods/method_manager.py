from typing import Callable, Tuple, Type, Dict

from federated_methods.fedavg import fedavg_load_state_dict, fedavg_aggregate_state_dict, fedavg_create_trainer, fedavg_memefficient_aggregate_state_dict, fedavg_memefficient_load_state_dict, fedavg_heterosimple_aggregate_state_dict
from federated_methods.sft import sft_load_state_dict
from federated_methods.task_id import task_id_create_trainer
from federated_methods.fedours import fedours_ema_distill_create_trainer, fedours_load_state_dict, OURS_set_state_dict, OURS_aggregate_state_dict, OURS_memefficient_aggregate_state_dict, fedours_memefficient_load_state_dict, fedsim_load_state_dict, fedours_include_load_state_dict, fedours_hetero_load_state_dict, fedours_self_load_state_dict

from federated_methods.fedpq import (feddualpq_load_state_dict, fedpq_load_state_dict, fedduallastpq_load_state_dict, 
                                     fedlastpq_load_state_dict, fedFLpq_load_state_dict, feddualFLpq_load_state_dict, 
                                     fedFMLpq_load_state_dict, feddualFMLpq_load_state_dict, feddualMultipq_load_state_dict,
                                     fedMultipq_load_state_dict, fedlastpq_tv_load_state_dict, fedMultipq_tv_load_state_dict,
                                     feddualMulti2pq_load_state_dict, fedMulti2pq_load_state_dict, fedMulti2pq_tv_load_state_dict,
                                     feddualMultipq_include_load_state_dict, fedBlockpq_load_state_dict, feddualMultipq_homoAgg_load_state_dict, feddualBlockpq_load_state_dict,
                                     feddualMulti05pq_homoAgg_load_state_dict,feddualMulti05pq_load_state_dict)

from federated_methods.fedavg_layerwiseloss import fedavg_layerwise_create_trainer
from federated_methods.fedours_layerwiseloss import fedours_layerwise_create_trainer
from federated_methods.fedours_pqgrad import fedours_pqgrad_create_trainer

def dummy_function(*args):
    return {}

def select_method(mode: str) -> Tuple[Callable, Callable, Callable, Callable, Dict]:
    extra_modules = {}
    if mode in ['sft', 'sft_only_B_train']:
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, sft_load_state_dict, fedavg_create_trainer, fedavg_aggregate_state_dict
    elif mode in ['fedavg', 'fedavg_only_B_train']:
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, fedavg_load_state_dict, fedavg_create_trainer, fedavg_aggregate_state_dict
    elif mode == 'fedsim_ours' or mode == 'fedsim_tv':
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, fedsim_load_state_dict, fedavg_create_trainer, fedavg_aggregate_state_dict
    elif mode in ['fedours', 'fedours_tv', 'fedours_excludemean', 'fedours_moe', 'fedours_only_B_train', 'fedours_tv_only_B_train']:
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, fedours_load_state_dict, fedours_ema_distill_create_trainer, OURS_aggregate_state_dict
    elif mode == 'fedours_include' or mode == 'fedours_tv_include' or mode == 'fedours_excludemean_include':
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, fedours_include_load_state_dict, fedours_ema_distill_create_trainer, OURS_aggregate_state_dict
    elif mode == 'fedours_self':
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, fedours_self_load_state_dict, fedours_ema_distill_create_trainer, OURS_aggregate_state_dict
    elif 'sft' in mode:
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, sft_load_state_dict, fedavg_create_trainer, fedavg_aggregate_state_dict
    elif mode in ['feddualpq','feddualpqfullfreeze','feddualpqfullfreeze_tv','feddualpqfreezeA','feddualpqfullfreezeA']:
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, feddualpq_load_state_dict, fedours_ema_distill_create_trainer, OURS_aggregate_state_dict
    elif mode =='fedpqfullfreeze' or mode == 'fedpqfullfreezeA' or mode == 'fedpqfreezeA':
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, fedpq_load_state_dict, fedavg_create_trainer, fedavg_aggregate_state_dict
    elif mode in ['feddualMultipqfreeze','feddualMultipqfullfreeze','feddualMultipqfullfreeze_tv','feddualMultipqfullfreeze_excludemean','feddualMultipqfullfreeze_moe',
                  'feddualMultipqfullfreezeA','feddualMultipqfullfreezeA_tv','feddualMultipqfullfreezeA_excludemean',
                  'feddualMultipqfreezeA','feddualMultipqfreezeA_excludemean',
                  'feddualMultipqfullfreeze256','feddualMultipqfullfreeze512','feddualMultipqfullfreeze1024',
                  'feddualMultipqfullfreeze256_tv','feddualMultipqfullfreeze512_tv','feddualMultipqfullfreeze1024_tv',
                  'feddualMultipqLILfullfreeze512','feddualMultipqLILfullfreeze1024',
                'feddualMultipqLILfullfreeze512_NL','feddualMultipqLILfullfreeze1024_NL',
                'feddualMultipfullfreeze',]:
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, feddualMultipq_load_state_dict, fedours_ema_distill_create_trainer, OURS_aggregate_state_dict
    elif mode in ['feddualMultipqfullfreeze_pqgrad', 'feddualMultipqfullfreeze_pqfisher']:
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, feddualMultipq_load_state_dict, fedours_pqgrad_create_trainer, OURS_aggregate_state_dict
    
    elif mode in ['feddualMulti05pqfullfreeze','feddualMulti05pqfullfreeze_excludemean',]:
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, feddualMulti05pq_load_state_dict, fedours_ema_distill_create_trainer, OURS_aggregate_state_dict
    
    elif mode =='feddualMultipqfullfreeze_include' or mode == 'feddualMultipqfullfreeze_tv_include' or mode == 'feddualMultipqfullfreeze_excludemean_include':
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, feddualMultipq_include_load_state_dict, fedours_ema_distill_create_trainer, OURS_aggregate_state_dict    
    
    elif mode =='feddualMulti2pqfullfreeze' or mode == 'feddualMulti2pqfullfreeze_tv' or mode == 'feddualMulti2pqfullfreeze_excludemean'\
        or mode =='feddualMulti2pqfullfreezeA' or mode == 'feddualMulti2pqfullfreezeA_tv' or mode == 'feddualMulti2pqfullfreezeA_excludemean'\
        or mode == 'feddualMulti2pqfreezeA' or mode == 'feddualMulti2pqfreezeA_excludemean':
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, feddualMulti2pq_load_state_dict, fedours_ema_distill_create_trainer, OURS_aggregate_state_dict

    elif mode in ['fedMultipqfullfreeze', 'fedMultipqfullfreezeA', 'fedMultipqfreezeA',
                  'fedMultipqfullfreeze256','fedMultipqfullfreeze512','fedMultipqfullfreeze1024']:
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, fedMultipq_load_state_dict, fedavg_create_trainer, OURS_aggregate_state_dict
    
    elif mode in ['fedBlock2pqfullfreeze', 'fedBlock4pqfullfreeze']:
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, fedBlockpq_load_state_dict, fedavg_create_trainer, OURS_aggregate_state_dict
    elif mode in ['feddualBlock2pqfullfreeze', 'feddualBlock4pqfullfreeze']:
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, feddualBlockpq_load_state_dict, fedours_ema_distill_create_trainer, OURS_aggregate_state_dict
    elif mode =='fedMultipqfullfreeze_tv' or mode =='fedMultipqfullfreeze_ours':
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, fedMultipq_tv_load_state_dict, fedavg_create_trainer, OURS_aggregate_state_dict
    elif mode =='fedMulti2pqfullfreeze' or mode =='fedMulti2pqfullfreezeA' or mode == 'fedMulti2pqfreezeA':
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, fedMulti2pq_load_state_dict, fedavg_create_trainer, OURS_aggregate_state_dict
    elif mode =='fedMulti2pqfullfreeze_tv' or mode =='fedMulti2pqfullfreeze_ours':
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, fedMulti2pq_tv_load_state_dict, fedavg_create_trainer, OURS_aggregate_state_dict
        
    elif mode == 'fedavg_hetero':
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, fedavg_load_state_dict, fedavg_create_trainer, fedavg_heterosimple_aggregate_state_dict
    elif mode == 'fedours_excludemean_hetero' or mode == 'fedours_hetero':
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, fedours_hetero_load_state_dict, fedours_ema_distill_create_trainer, OURS_aggregate_state_dict
    elif mode in ['feddualMultipqfullfreeze_homoAgg', 'feddualMultipqfullfreeze_excludemean_homoAgg']:
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, feddualMultipq_homoAgg_load_state_dict, fedours_ema_distill_create_trainer, OURS_aggregate_state_dict
    
    elif mode in ['feddualMulti05pqfullfreeze_homoAgg', 'feddualMulti05pqfullfreeze_excludemean_homoAgg']:
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, feddualMulti05pq_homoAgg_load_state_dict, fedours_ema_distill_create_trainer, OURS_aggregate_state_dict
    
    elif mode in ['fedMultipqfullfreeze_distill', 'fedMultipqfullfreeze_Taskloss', 'fedMultipqfullfreeze_distillTaskloss']:
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, fedMultipq_load_state_dict, fedavg_layerwise_create_trainer, OURS_aggregate_state_dict
    elif mode in ['feddualMultipqfullfreeze_distill', 'feddualMultipqfullfreeze_Taskloss', 'feddualMultipqfullfreeze_distillTaskloss', 'feddualMultipqfullfreeze_KLloss', 'feddualMultipqfullfreeze_distillKLloss',
                  'feddualMultipqfullfreeze256_Taskloss', 'feddualMultipqfullfreeze512_Taskloss','feddualMultipqfullfreeze1024_Taskloss',
                    'feddualMultipqfullfreeze256_KLloss', 'feddualMultipqfullfreeze512_KLloss','feddualMultipqfullfreeze1024_KLloss',
                    'feddualMultipqfullfreeze256_distill', 'feddualMultipqfullfreeze512_distill','feddualMultipqfullfreeze1024_distill',
                    'feddualMultipqfullfreeze256_distillTaskloss', 'feddualMultipqfullfreeze512_distillTaskloss','feddualMultipqfullfreeze1024_distillTaskloss']:
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, feddualMultipq_load_state_dict, fedours_layerwise_create_trainer, OURS_aggregate_state_dict
    
    elif mode =='fedavg_memefficient':
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, fedavg_memefficient_load_state_dict, fedavg_create_trainer, fedavg_memefficient_aggregate_state_dict
    elif mode == 'fedours_memefficient':
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, fedours_memefficient_load_state_dict, fedours_ema_distill_create_trainer, OURS_memefficient_aggregate_state_dict 
    else:
        raise NotImplementedError(mode)
    return set_state_dict, load_state_dict, create_trainer, aggregate_state_dict, extra_modules
