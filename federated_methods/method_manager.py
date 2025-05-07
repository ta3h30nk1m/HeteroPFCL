from typing import Callable, Tuple, Type, Dict

from federated_methods.fedavg import fedavg_load_state_dict, fedavg_aggregate_state_dict, fedavg_create_trainer, fedavg_memefficient_load_state_dict
from federated_methods.sft import sft_load_state_dict, fedper_load_state_dict
from federated_methods.task_id import task_id_create_trainer
from federated_methods.fedours import fedours_ema_distill_create_trainer, fedours_load_state_dict, OURS_set_state_dict, OURS_aggregate_state_dict, OURS_memefficient_aggregate_state_dict, fedours_memefficient_load_state_dict, fedsim_load_state_dict, fedours_include_load_state_dict, fedours_hetero_load_state_dict, fedours_self_load_state_dict

from federated_methods.fedpq import (feddualpq_load_state_dict,  feddualMultipq_load_state_dict,
                                     fedMultipq_load_state_dict, fedMultipq_tv_load_state_dict,
                                     feddualMultipq_include_load_state_dict, fedBlockpq_load_state_dict, feddualMultipq_homoAgg_load_state_dict, feddualBlockpq_load_state_dict,feddualMultipq_include_homoAgg_load_state_dict,
                                     feddualMulti05pq_homoAgg_load_state_dict,feddualMulti05pq_load_state_dict,feddualMulti05pq_include_load_state_dict,feddualMulti05pq_include_homoAgg_load_state_dict,
                                     feddualMulti05pq_homoAggOnly_load_state_dict,feddualMultipq_homoAggOnly_load_state_dict, feddualOptimalpq_load_state_dict,
                                     fedMultipq_HomoAgg_load_state_dict, fedMultipq_HomoAggOnly_load_state_dict,
                                     feddualMulti2pq_load_state_dict, feddualMulti2pq_homoAgg_load_state_dict,
                                     fedMulti05pq_load_state_dict,fedMulti05pq_HomoAgg_load_state_dict,fedMulti05pq_HomoAggOnly_load_state_dict,
                                     feddualMulti05pq_homoAgg_Normalize_load_state_dict,feddualMultipq_homoAgg_Normalize_load_state_dict,
                                     feddualMultipqfull_homoAgg_load_state_dict,feddualMulti2pqfull_homoAgg_load_state_dict,
                                     )

from federated_methods.sft_layerwiseloss import sft_layerwise_create_trainer
from federated_methods.fedavg_layerwiseloss import fedavg_layerwise_create_trainer
from federated_methods.fedours_layerwiseloss import fedours_layerwise_create_trainer
from federated_methods.fedours_pqgrad import fedours_pqgrad_create_trainer

from federated_methods.feddat import feddat_create_trainer, feddat_hetero_load_state_dict, feddat_aggregate_state_dict, feddat_hetero_pqlora_load_state_dict
from federated_methods.feddistill import Distillation_aggregate_state_dict
from federated_methods.perada import perada_create_trainer
from federated_methods.fedsim import fedsim_create_trainer
from federated_methods.fdlora import fdlora_aggregate_state_dict, fdlora_blockwise_aggregate_state_dict
from federated_methods.takfl import TAKFL_aggregate_state_dict
from federated_methods.fedmkt import FEDMKT_aggregate_state_dict
from federated_methods.feddpa import feddpa_create_trainer

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
    elif mode in ['fedours', 'fedours_tv', 'fedours_excludemean', 'fedours_moe', 'fedours_only_B_train', 'fedours_tv_only_B_train',
                  'fedquad_grad', 'fedquad_excludemean','fedquad_grad_moe', 'fedquad_excludemean_moe',
                  'fedhexa_grad', 'fedhexa_grad_moe']:
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, fedours_load_state_dict, fedours_ema_distill_create_trainer, OURS_aggregate_state_dict
    elif mode in ['fedours_include','fedours_tv_include','fedours_excludemean_include','fedours_include_moe', 'fedours_excludemean_include_moe',
                  'fedquad_grad_include', 'fedquad_excludemean_include', 'fedquad_grad_include_moe', 'fedquad_excludemean_include_moe',
                  'fedhexa_grad_include', 'fedhexa_grad_include_moe'] :
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, fedours_include_load_state_dict, fedours_ema_distill_create_trainer, OURS_aggregate_state_dict
    
    elif mode in ['fedours_pqgrad_moe']:
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, fedours_load_state_dict, fedours_pqgrad_create_trainer, OURS_aggregate_state_dict
    
    elif mode == 'fedours_self':
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, fedours_self_load_state_dict, fedours_ema_distill_create_trainer, OURS_aggregate_state_dict
    elif mode == 'fedMultipqfullfreeze_sft_Taskloss':
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, sft_load_state_dict, sft_layerwise_create_trainer, fedavg_aggregate_state_dict
    elif 'sft' in mode:
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, sft_load_state_dict, fedavg_create_trainer, fedavg_aggregate_state_dict
    
    elif mode in ['feddistill']:
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, fedavg_load_state_dict, fedavg_create_trainer, Distillation_aggregate_state_dict
    
    elif mode in ['feddat']:
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, feddat_hetero_load_state_dict, feddat_create_trainer, feddat_aggregate_state_dict
    elif mode in ['feddat_Multipqfullfreeze', 'feddat_Multi05pqfullfreeze']:
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, feddat_hetero_pqlora_load_state_dict, feddat_create_trainer, feddat_aggregate_state_dict
    
    elif mode in ['fedsim','fedsim_hetero','fedsim_feddualMultipqfullfreeze_homoAgg', 'fedsim_feddualMulti05pqfullfreeze_homoAgg']:
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, fedper_load_state_dict, fedsim_create_trainer, fedavg_aggregate_state_dict

    elif mode in ['ditto','ditto_feddualMultipqfullfreeze_homoAgg', 'ditto_feddualMulti05pqfullfreeze_homoAgg']:
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, fedper_load_state_dict, perada_create_trainer, fedavg_aggregate_state_dict
    
    elif mode in ['feddpa','feddpa_feddualMultipqfullfreeze_homoAgg', 'feddpa_feddualMulti05pqfullfreeze_homoAgg']:
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, fedper_load_state_dict, feddpa_create_trainer, fedavg_aggregate_state_dict
    elif mode in ['fdlora']:
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, fedavg_load_state_dict, fedavg_create_trainer, fdlora_aggregate_state_dict
    elif mode in ['fdlora_fedMultipqfullfreeze_homoAgg', 'fdlora_fedMulti05pqfullfreeze_homoAgg']:
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, fedavg_load_state_dict, fedavg_create_trainer, fdlora_blockwise_aggregate_state_dict
    
    elif mode in ['perada','perada_feddualMultipqfullfreeze','perada_feddualMulti05pqfullfreeze']:
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, fedper_load_state_dict, perada_create_trainer, Distillation_aggregate_state_dict
    
    elif mode in ['takfl', 'takfl_fedMultipqfullfreeze_homoAgg', 'takfl_fedMulti05pqfullfreeze_homoAgg']:
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, fedavg_load_state_dict, fedavg_create_trainer, TAKFL_aggregate_state_dict
    
    elif mode in ['fedmkt', 'fedmkt_fedMultipqfullfreeze_homoAgg', 'fedmkt_fedMulti05pqfullfreeze_homoAgg']:
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, sft_load_state_dict, fedavg_create_trainer, FEDMKT_aggregate_state_dict
    
    elif mode in ['feddualpq','feddualpqfullfreeze','feddualpqfullfreeze_tv','feddualpqfreezeA','feddualpqfullfreezeA']:
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, feddualpq_load_state_dict, fedours_ema_distill_create_trainer, OURS_aggregate_state_dict
    elif mode =='fedpqfullfreeze' or mode == 'fedpqfullfreezeA' or mode == 'fedpqfreezeA':
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, fedpq_load_state_dict, fedavg_create_trainer, fedavg_aggregate_state_dict
    elif mode in ['feddualMultipqfreeze','feddualMultipqfullfreeze','feddualMultipqfullfreeze_tv','feddualMultipqfullfreeze_excludemean','feddualMultipqfullfreeze_moe',
                  'feddualMultipqfullfreezeA','feddualMultipqfullfreezeA_tv','feddualMultipqfullfreezeA_excludemean',
                  'feddualMultipqfreezeA','feddualMultipqfreezeA_excludemean',
                  'feddualMultipqfullfreeze256','feddualMultipqfullfreeze512','feddualMultipqfullfreeze1024',
                  'feddualMultipqfullfreeze256_tv','feddualMultipqfullfreeze512_tv','feddualMultipqfullfreeze1024_tv',
                  'feddualMultipqLILfullfreeze512','feddualMultipqLILfullfreeze1024','feddualMultipqLILfullfreeze128','feddualMultipqLILfullfreeze256',
                'feddualMultipqLILfullfreeze512_NL','feddualMultipqLILfullfreeze1024_NL',
                'feddualMultipfullfreeze',
                'fedquadMultipqfullfreeze','fedquadMultipqfullfreeze_moe',
                'feddualMultipqfull','feddualMultipqfullfreezeA2','feddualMultipqfull2',
                ]:
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, feddualMultipq_load_state_dict, fedours_ema_distill_create_trainer, OURS_aggregate_state_dict
    elif mode in ['feddualMultipqfullfreeze_homoAgg', 'feddualMultipqfullfreeze_excludemean_homoAgg', 'feddualMultipqfullfreeze_homoAgg_moe','feddualMultipfullfreeze_homoAgg_moe',
                  'fedquadMultipqfullfreeze_homoAgg','fedquadMultipqfullfreeze_homoAgg_moe']:
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, feddualMultipq_homoAgg_load_state_dict, fedours_ema_distill_create_trainer, OURS_aggregate_state_dict
    elif mode in ['feddualMultipqfullfreezeA_homoAgg_moe','feddualMultipqfullfreezeB_homoAgg_moe','feddualMultipqfull_homoAgg_moe','feddualMultipqfull_homoAgg_moe2','feddualMultipqfullfreezeA_homoAgg_moe2',]:
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, feddualMultipqfull_homoAgg_load_state_dict, fedours_ema_distill_create_trainer, OURS_aggregate_state_dict
    elif mode in ['feddualMultipqfullfreeze_homoAgg_normalize_moe']:
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, feddualMultipq_homoAgg_Normalize_load_state_dict, fedours_ema_distill_create_trainer, OURS_aggregate_state_dict
    elif mode in ['feddualMultipqfullfreeze_include','feddualMultipqfullfreeze_tv_include', 'feddualMultipqfullfreeze_excludemean_include','feddualMultipqfullfreeze_include_moe',
                  'fedquadMultipqfullfreeze_include', 'fedquadMultipqfullfreeze_include_moe',]:
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, feddualMultipq_include_load_state_dict, fedours_ema_distill_create_trainer, OURS_aggregate_state_dict
    elif mode in ['feddualMultipqfullfreeze_include_homoAgg', 'feddualMultipqfullfreeze_excludemean_include_homoAgg','feddualMultipqfullfreeze_include_homoAgg_moe',
                  'fedquadMultipqfullfreeze_include_homoAgg', 'fedquadMultipqfullfreeze_include_homoAgg_moe']:
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, feddualMultipq_include_homoAgg_load_state_dict, fedours_ema_distill_create_trainer, OURS_aggregate_state_dict
    elif mode in ['feddualMulti05pqfullfreeze','feddualMulti05pqfullfreeze_excludemean','feddualMulti05pqfullfreeze_moe',
                  'fedquadMulti05pqfullfreeze','fedquadMulti05pqfullfreeze_moe',]:
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, feddualMulti05pq_load_state_dict, fedours_ema_distill_create_trainer, OURS_aggregate_state_dict
    elif mode in ['feddualMulti05pqfullfreeze_homoAgg', 'feddualMulti05pqfullfreeze_excludemean_homoAgg','feddualMulti05pqfullfreeze_homoAgg_moe','feddualMulti05pfullfreeze_homoAgg_moe',
                  'fedquadMulti05pqfullfreeze_homoAgg','fedquadMulti05pqfullfreeze_homoAgg_moe',]:
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, feddualMulti05pq_homoAgg_load_state_dict, fedours_ema_distill_create_trainer, OURS_aggregate_state_dict
    elif mode in ['feddualMulti05pqfullfreeze_homoAgg_normalize_moe']:
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, feddualMulti05pq_homoAgg_Normalize_load_state_dict, fedours_ema_distill_create_trainer, OURS_aggregate_state_dict
    elif mode in ['feddualMulti05pqfullfreeze_include','feddualMulti05pqfullfreeze_excludemean_include','feddualMulti05pqfullfreeze_include_moe',
                  'fedquadMulti05pqfullfreeze_include','fedquadMulti05pqfullfreeze_include_moe',]:
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, feddualMulti05pq_include_load_state_dict, fedours_ema_distill_create_trainer, OURS_aggregate_state_dict
    elif mode in ['feddualMulti05pqfullfreeze_include_homoAgg', 'feddualMulti05pqfullfreeze_excludemean_include_homoAgg','feddualMulti05pqfullfreeze_include_homoAgg_moe',
                  'fedquadMulti05pqfullfreeze_include_homoAgg', 'fedquadMulti05pqfullfreeze_include_homoAgg_moe']:
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, feddualMulti05pq_include_homoAgg_load_state_dict, fedours_ema_distill_create_trainer, OURS_aggregate_state_dict
    elif mode in ['fedMultipqfullfreeze', 'fedMultipqfullfreezeA', 'fedMultipqfreezeA', 'feddualMultipq_freezeA_trainB_weightnormP',
                  'fedMultipqfullfreeze256','fedMultipqfullfreeze512','fedMultipqfullfreeze1024']:
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, fedMultipq_load_state_dict, fedavg_create_trainer, OURS_aggregate_state_dict
    
    elif mode in ['fedBlock2pqfullfreeze', 'fedBlock4pqfullfreeze']:
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, fedBlockpq_load_state_dict, fedavg_create_trainer, OURS_aggregate_state_dict
    elif mode in ['feddualBlock2pqfullfreeze', 'feddualBlock4pqfullfreeze']:
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, feddualBlockpq_load_state_dict, fedours_ema_distill_create_trainer, OURS_aggregate_state_dict
    elif mode =='fedMultipqfullfreeze_tv' or mode =='fedMultipqfullfreeze_ours':
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, fedMultipq_tv_load_state_dict, fedavg_create_trainer, OURS_aggregate_state_dict
        
    elif mode == 'fedavg_hetero':
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, fedavg_load_state_dict, fedavg_create_trainer, fedavg_aggregate_state_dict
    elif mode == 'fedours_excludemean_hetero' or mode == 'fedours_hetero' or mode == 'fedours_hetero_moe':
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, fedours_hetero_load_state_dict, fedours_ema_distill_create_trainer, OURS_aggregate_state_dict
    elif mode in ['feddualMulti05pqfullfreeze_homoAggOnly']:
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, feddualMulti05pq_homoAggOnly_load_state_dict, fedours_ema_distill_create_trainer, OURS_aggregate_state_dict
    elif mode in ['feddualMultipqfullfreeze_homoAggOnly','feddualMultipqfullfreeze_homoAggOnly_moe', 'feddualMultipqfull_homoAggOnly_moe']:
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, feddualMultipq_homoAggOnly_load_state_dict, fedours_ema_distill_create_trainer, OURS_aggregate_state_dict
    elif mode in ['fedMultipqfullfreeze_distill', 'fedMultipqfullfreeze_Taskloss', 'fedMultipqfullfreeze_distillTaskloss',]:
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, fedMultipq_load_state_dict, fedavg_layerwise_create_trainer, OURS_aggregate_state_dict
    elif mode in ['feddualMultipqfullfreeze_distill', 'feddualMultipqfullfreeze_Taskloss', 'feddualMultipqfullfreeze_distillTaskloss', 'feddualMultipqfullfreeze_KLloss', 'feddualMultipqfullfreeze_distillKLloss',
                  'feddualMultipqfullfreeze256_Taskloss', 'feddualMultipqfullfreeze512_Taskloss','feddualMultipqfullfreeze1024_Taskloss',
                    'feddualMultipqfullfreeze256_KLloss', 'feddualMultipqfullfreeze512_KLloss','feddualMultipqfullfreeze1024_KLloss',
                    'feddualMultipqfullfreeze256_distill', 'feddualMultipqfullfreeze512_distill','feddualMultipqfullfreeze1024_distill',
                    'feddualMultipqfullfreeze256_distillTaskloss', 'feddualMultipqfullfreeze512_distillTaskloss','feddualMultipqfullfreeze1024_distillTaskloss',
                    'feddualMultipqLILfullfreeze512_Taskloss','feddualMultipqLILfullfreeze512_KLloss','feddualMultipqLILfullfreeze512_distillTaskloss']:
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, feddualMultipq_load_state_dict, fedours_layerwise_create_trainer, OURS_aggregate_state_dict
    
    elif mode in ['feddualMultipqfullfreeze_homoAgg_moe_Taskloss','feddualMultipqfullfreeze_homoAgg_moe_KLloss']:
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, feddualMultipq_homoAgg_load_state_dict, fedours_layerwise_create_trainer, OURS_aggregate_state_dict
    
    elif mode in 'feddualMultipqWNfullfreeze':
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, feddualMultipq_load_state_dict, fedours_layerwise_create_trainer, OURS_aggregate_state_dict
    elif mode in ['feddualOptimal2pqfullfreeze','feddualOptimal4pqfullfreeze','feddualOptimal8pqfullfreeze']:
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, feddualOptimalpq_load_state_dict, fedours_ema_distill_create_trainer, OURS_aggregate_state_dict
        
    elif mode in ['feddualMulti2pqfullfreeze_front', 'feddualMulti2pqfullfreeze_back', 'feddualMulti2pqfullfreeze_back_moe','feddualMulti2pfullfreeze_back','feddualMulti2pqfullfreezeA_back2','feddualMulti2pqfull_back2',]:
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, feddualMulti2pq_load_state_dict, fedours_ema_distill_create_trainer, OURS_aggregate_state_dict
    elif mode in ['feddualMulti2pqfullfreeze_back_homoAgg', 'feddualMulti2pqfullfreeze_back_homoAgg_moe',]:
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, feddualMulti2pq_homoAgg_load_state_dict, fedours_ema_distill_create_trainer, OURS_aggregate_state_dict
    elif mode in ['feddualMulti2pqfullfreezeA_back_homoAgg_moe', 'feddualMulti2pqfull_back_homoAgg_moe2',]:
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, feddualMulti2pqfull_homoAgg_load_state_dict, fedours_ema_distill_create_trainer, OURS_aggregate_state_dict
    
    elif mode in ['fedMultipqfullfreeze_homoAgg']:
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, fedMultipq_HomoAgg_load_state_dict, fedavg_create_trainer, OURS_aggregate_state_dict
    elif mode in ['fedMultipqfullfreeze_homoAggOnly']:
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, fedMultipq_HomoAggOnly_load_state_dict, fedavg_create_trainer, OURS_aggregate_state_dict
    elif mode in ['fedMulti05pqfullfreeze']:
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, fedMulti05pq_load_state_dict, fedavg_create_trainer, OURS_aggregate_state_dict
    elif mode in ['fedMulti05pqfullfreeze_homoAgg']:
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, fedMulti05pq_HomoAgg_load_state_dict, fedavg_create_trainer, OURS_aggregate_state_dict
    elif mode in ['fedMulti05pqfullfreeze_homoAggOnly']:
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, fedMulti05pq_HomoAggOnly_load_state_dict, fedavg_create_trainer, OURS_aggregate_state_dict
    
    else:
        raise NotImplementedError(mode)
    return set_state_dict, load_state_dict, create_trainer, aggregate_state_dict, extra_modules
