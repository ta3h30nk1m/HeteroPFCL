from typing import Callable, Tuple, Type, Dict

from federated_methods.fedavg import fedavg_load_state_dict, fedavg_aggregate_state_dict, fedavg_create_trainer
from federated_methods.sft import sft_load_state_dict
from federated_methods.fedper import fedper_set_state_dict, fedper_load_state_dict
from federated_methods.feddat import feddat_set_state_dict, feddat_create_trainer, feddat_aggregate_state_dict
from federated_methods.fedprox import fedprox_set_state_dict, fedprox_create_trainer
from federated_methods.pfedme import pfedme_set_state_dict, pfedme_create_trainer, pfedme_aggregate_state_dict
from federated_methods.fedsim import fedsim_set_state_dict, fedsim_create_trainer
from federated_methods.ditto import ditto_create_trainer, ditto_set_state_dict
from federated_methods.apfl import apfl_create_trainer
from federated_methods.task_id import task_id_create_trainer
from federated_methods.lae import LAE_create_trainer
from federated_methods.ditto_lae import ditto_lae_set_state_dict, ditto_lae_create_trainer
from federated_methods.ours_generator_ema_distill import OURS_set_state_dict, OURS_aggregate_state_dict, OURS_GEN_ema_distill_create_trainer, OURS_GEN_load_state_dict
from federated_methods.ours_generator_ema_ewc import OURS_GEN_ema_ewc_create_trainer

from federated_methods.fedours import fedours_ema_distill_create_trainer, fedours_load_state_dict

def dummy_function(*args):
    return {}

def select_method(mode: str) -> Tuple[Callable, Callable, Callable, Callable, Dict]:
    extra_modules = {}
    if mode == 'sft':
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, sft_load_state_dict, fedavg_create_trainer, fedavg_aggregate_state_dict
    elif mode == 'fedavg':
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, fedavg_load_state_dict, fedavg_create_trainer, fedavg_aggregate_state_dict
    elif mode =='ditto':
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = ditto_set_state_dict, fedper_load_state_dict, ditto_create_trainer, fedavg_aggregate_state_dict
        
    elif mode =='fedper':
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = fedper_set_state_dict, fedper_load_state_dict, fedavg_create_trainer, fedavg_aggregate_state_dict
        
    elif mode =='fedsim':
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = fedsim_set_state_dict, fedper_load_state_dict, fedsim_create_trainer, fedavg_aggregate_state_dict
    
    elif mode =='feddat':
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = feddat_set_state_dict, fedper_load_state_dict, feddat_create_trainer, feddat_aggregate_state_dict
    
    elif mode == 'L2P_T2' or mode =='L2P2' \
        or mode == 'DAP' or mode == 'DAP_T' \
        or mode == 'CodaPrompt' or mode == 'CodaPrompt_T':
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, sft_load_state_dict, task_id_create_trainer, fedavg_aggregate_state_dict
    
    elif mode =='L2P_T2_FedAvg' or mode =='L2P2_FedAvg' \
        or mode == 'DAP_FedAvg' or mode == 'DAP_T_FedAvg' \
        or mode == 'CodaPrompt_FedAvg' or mode =='CodaPrompt_T_FedAvg':
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, fedavg_load_state_dict, task_id_create_trainer, fedavg_aggregate_state_dict
    
    elif mode =='L2P_T2_FedPer' or mode =='L2P2_FedPer' \
        or mode == 'DAP_FedPer' or mode == 'DAP_T_FedPer' \
        or mode == 'CodaPrompt_FedPer' or mode =='CodaPrompt_T_FedPer':
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = fedper_set_state_dict, fedper_load_state_dict, task_id_create_trainer, fedavg_aggregate_state_dict
    
    elif mode =='L2P_T2_Ditto' or mode =='L2P2_Ditto' \
        or mode == 'DAP_Ditto' or mode == 'DAP_T_Ditto' \
        or mode == 'CodaPrompt_Ditto' or mode =='CodaPrompt_T_Ditto':
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = ditto_set_state_dict, fedper_load_state_dict, ditto_create_trainer, fedavg_aggregate_state_dict
    
    elif mode == 'LAE':
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, sft_load_state_dict, LAE_create_trainer, fedavg_aggregate_state_dict
    elif mode == 'LAE_FedAvg':
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, fedavg_load_state_dict, LAE_create_trainer, fedavg_aggregate_state_dict
    elif mode == 'LAE_FedPer':
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = fedper_set_state_dict, fedper_load_state_dict, LAE_create_trainer, fedavg_aggregate_state_dict
    elif mode == 'LAE_Ditto':
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = ditto_lae_set_state_dict, fedper_load_state_dict, ditto_lae_create_trainer, fedavg_aggregate_state_dict

    
    elif mode == 'L2P2_FedDAT' or mode =='L2P_T_FedDAT':
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = feddat_set_state_dict, fedper_load_state_dict, feddat_create_trainer, feddat_aggregate_state_dict
        
    # elif mode =='EvoPrompt' or mode == 'EvoPrompt_T':
    #     set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, sft_load_state_dict, OT_create_trainer, fedavg_aggregate_state_dict
    
    elif mode =='ours_generator':
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = OURS_set_state_dict, OURS_GEN_load_state_dict, OURS_GEN_ema_distill_create_trainer, OURS_aggregate_state_dict
    elif mode == 'ours_generator2':
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = OURS_set_state_dict, OURS_GEN_load_state_dict, OURS_GEN_ema_distill_create_trainer, OURS_aggregate_state_dict
    
    elif mode == 'fedours' or mode == 'fedours_moe':
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = OURS_set_state_dict, fedours_load_state_dict, fedours_ema_distill_create_trainer, OURS_aggregate_state_dict    
    elif mode =='L2P_T2_fedours':
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = OURS_set_state_dict, fedours_load_state_dict, fedours_ema_distill_create_trainer, OURS_aggregate_state_dict    
    
    else:
        raise NotImplementedError(mode)
    return set_state_dict, load_state_dict, create_trainer, aggregate_state_dict, extra_modules
