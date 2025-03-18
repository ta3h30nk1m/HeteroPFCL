from transformers import LlamaForCausalLM, AutoModel, AutoModelForCausalLM
import torch
from torch import nn
from typing import List, Optional, Tuple, Union
from transformers.utils import logging

from models.dual_ia3.dual_ia3_layer import DualIA3Layer
from models.duallora.dualloralayer import DualLoraLayer
from models.duallora_moe.dualmoeloralayer import DualMOELoraLayer
from models.dual_pqlora.dual_pqloralayer import PQLoraLayer
from models.dual_pqlora_freeze.dual_pqloralayer_freeze import PQLoraFreezeLayer
from models.dual_pqlora_freezeA.dual_pqloralayer_freezeA import PQLoraFreezeALayer
from models.dual_pqlora_freeze_full.dual_pqloralayer_freeze_full import PQLoraFullFreezeLayer
from models.dual_pqlora_freeze_full_moe.dual_pqloralayer_freeze_full_moe import PQMOELoraFullFreezeLayer
from models.dual_pqlora_freezeA_full.dual_pqloralayer_freezeA_full import PQLoraFullFreezeALayer
from models.dual_pqlora_LIL_freeze_full.dual_pqloraLILlayer_freeze_full import PQLoraLILFullFreezeLayer
from models.dual_plora_freeze_full.dual_ploralayer_freeze_full import PLoraFullFreezeLayer
logger = logging.get_logger(__name__)

class CustomLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        
        self.active_state = 'gate'
    
    def set_state(self, state):
        assert state in ['lora1', 'lora2', 'gate'], state
        self.active_state = state
        
        for name, module in self.named_modules():
            if isinstance(module, DualLoraLayer) or isinstance(module, DualIA3Layer) or isinstance(module, DualMOELoraLayer) \
                or isinstance(module, PQLoraLayer) or isinstance(module, PQLoraFreezeLayer) or isinstance(module, PQLoraFreezeALayer) or isinstance(module, PQLoraLILFullFreezeLayer) \
                or isinstance(module, PQLoraFullFreezeLayer) or isinstance(module, PQLoraFullFreezeALayer) or isinstance(module, PQMOELoraFullFreezeLayer) or isinstance(module, PLoraFullFreezeLayer):
                module.set_state(state)

    def activate_all(self):
        for name, module in self.named_modules():
            if isinstance(module, DualLoraLayer) or isinstance(module, DualIA3Layer) or isinstance(module, DualMOELoraLayer) \
                or isinstance(module, PQLoraLayer) or isinstance(module, PQLoraFreezeLayer) or isinstance(module, PQLoraFreezeALayer) or isinstance(module, PQLoraLILFullFreezeLayer) \
                or isinstance(module, PQLoraFullFreezeLayer) or isinstance(module, PQLoraFullFreezeALayer) or isinstance(module, PQMOELoraFullFreezeLayer) or isinstance(module, PLoraFullFreezeLayer):
                module.activate_all()

    def activate_lora1(self):
        for name, module in self.named_modules():
            if isinstance(module, DualLoraLayer) or isinstance(module, DualIA3Layer) or isinstance(module, DualMOELoraLayer) \
                or isinstance(module, PQLoraLayer) or isinstance(module, PQLoraFreezeLayer) or isinstance(module, PQLoraFreezeALayer) or isinstance(module, PQLoraLILFullFreezeLayer) \
                or isinstance(module, PQLoraFullFreezeLayer) or isinstance(module, PQLoraFullFreezeALayer) or isinstance(module, PQMOELoraFullFreezeLayer) or isinstance(module, PLoraFullFreezeLayer):
                module.activate_lora1()
    
    def activate_lora2(self):
        for name, module in self.named_modules():
            if isinstance(module, DualLoraLayer) or isinstance(module, DualIA3Layer) or isinstance(module, DualMOELoraLayer) \
                or isinstance(module, PQLoraLayer) or isinstance(module, PQLoraFreezeLayer) or isinstance(module, PQLoraFreezeALayer) or isinstance(module, PQLoraLILFullFreezeLayer) \
                or isinstance(module, PQLoraFullFreezeLayer) or isinstance(module, PQLoraFullFreezeALayer) or isinstance(module, PQMOELoraFullFreezeLayer) or isinstance(module, PLoraFullFreezeLayer):
                module.activate_lora2()
        