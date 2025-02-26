import torch
from torch import nn
import math
import os
import logging
import transformers
import models.llava.conversation as conversation_lib_llava
from peft.tuners.lora import LoraLayer

from transformers import AutoConfig, AutoProcessor, AutoModelForImageTextToText, LlavaForConditionalGeneration

from models.llava.llava_multi import LlavaMultiForConditionalGeneration
import copy
ACCESS_TOKEN = "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"

def get_VLMmodel(model_args, training_args, bnb_model_from_pretrained_args, data_args):
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
    attn_implementation = "flash_attention_2"

    # load tokenizer
    # for llava
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    if tokenizer.unk_token is not None and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
    
    if model_args.model_type == 'llama3-8b':
        tokenizer.pad_token = tokenizer.eos_token
        
    if training_args.is_eval:
        tokenizer.padding_side = "left"
    
    processor = AutoProcessor.from_pretrained(model_args.model_name_or_path)
    
    if 'llava' in model_args.model_name_or_path.lower() or 'vl' in model_args.model_name_or_path.lower():
        model = LlavaMultiForConditionalGeneration.from_pretrained( # LlavaForConditionalGeneration
            model_args.model_name_or_path,
            torch_dtype=compute_dtype,
            use_flash_attention_2=True
        )
    else:
        model = AutoModelForImageTextToText.from_pretrained( # LlavaForConditionalGeneration
            model_args.model_name_or_path,
            torch_dtype=compute_dtype,
            use_flash_attention_2=True
        )
        
    model.config.use_cache = False
    model.vision_tower.requires_grad_(False)
    
    model = model.to(training_args.device)
    
    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.bits == 16:
        if training_args.bf16:
            model.to(torch.bfloat16)
        if training_args.fp16:
            model.to(torch.float16)
    
    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
            exclude_modules=r".*vision_tower.*|.*multi_modal_projector.*", 
        )
        
        if training_args.mode in ['fedsim', 'apfl', 'ditto', 'fedours', 'fedours_tv', 'fedours_only_B_train', 'fedours_tv_only_B_train', 'fedours_excludemean',
                                  'fedours_include', 'fedours_tv_include', 'fedours_excludemean_include', 'fedours_excludemean_hetero','fedours_hetero', 'feddat']:
            from models.duallora.dualloramodel import DualLoraModel
            from peft.peft_model import PEFT_TYPE_TO_MODEL_MAPPING
            PEFT_TYPE_TO_MODEL_MAPPING['DUALLORA'] = DualLoraModel
            lora_config.peft_type = 'DUALLORA'
        if training_args.mode in ['fedours_moe']:
            from models.duallora_moe.dualmoeloramodel import DualMOELoraModel
            from peft.peft_model import PEFT_TYPE_TO_MODEL_MAPPING
            PEFT_TYPE_TO_MODEL_MAPPING['DUALMOELORA'] = DualMOELoraModel
            lora_config.peft_type = 'DUALMOELORA'
        elif training_args.mode in ['fedpq_sft', 'fedpq', 'fedlastpq', 'fedFLpq', 'fedFMLpq', 'fedlastpqfreeze', 'fedFLpqfreeze', 'fedMultipqfreezeA', 'fedMulti2pqfreezeA', 'fedMultipqfreezeA_sft', 'fedpqfreezeA_sft', 'fedpqfreezeA']:
            from models.pqlora.pqloramodel import PQLoraModel
            from peft.peft_model import PEFT_TYPE_TO_MODEL_MAPPING
            PEFT_TYPE_TO_MODEL_MAPPING['PQLORA'] = PQLoraModel
            lora_config.peft_type = 'PQLORA'
        
        elif training_args.mode in ['fedpqfullfreeze_sft', 'fedpqfullfreeze', 'fedpqfullfreezeA_sft', 'fedpqfullfreezeA',
                                    'fedlastpqfullfreeze_sft', 'fedlastpqfullfreeze', 'fedlastpqfullfreeze_tv', 'fedlastpqfullfreeze_ours', 
                                    'fedMultipqfullfreeze_sft', 'fedMultipqfullfreeze', 'fedMultipqfullfreeze_tv', 'fedMultipqfullfreeze_ours',
                                    'fedMulti2pqfullfreeze_sft', 'fedMulti2pqfullfreeze', 'fedMulti2pqfullfreeze_tv', 'fedMulti2pqfullfreeze_ours',
                                    'fedMulti2pqfullfreezeA','fedMultipqfullfreezeA', 'fedMultipqfullfreezeA_sft',
                                    'fedMultipqfullfreeze256_sft','fedMultipqfullfreeze256',
                                    'fedMultipqfullfreeze512_sft','fedMultipqfullfreeze512',
                                    'fedMultipqfullfreeze1024_sft','fedMultipqfullfreeze1024',
                                    'fedBlock2pqfullfreeze', 'fedBlock2pqfullfreeze_sft',
                                    'fedBlock4pqfullfreeze', 'fedBlock4pqfullfreeze_sft',
                                    'fedMultipqfullfreeze_distill', 'fedMultipqfullfreeze_Taskloss', 'fedMultipqfullfreeze_distillTaskloss'
                                     ]:
            from models.pqlora_full.pqloramodel_full import PQLoraModel
            from peft.peft_model import PEFT_TYPE_TO_MODEL_MAPPING
            PEFT_TYPE_TO_MODEL_MAPPING['PQFullLORA'] = PQLoraModel
            lora_config.peft_type = 'PQFullLORA'
        elif training_args.mode in ['feddualpq', 'fedduallastpq', 'feddualFLpq', 'feddualFMLpq']:
            from models.dual_pqlora.dual_pqloramodel import Dual_PQLoraModel
            from peft.peft_model import PEFT_TYPE_TO_MODEL_MAPPING
            PEFT_TYPE_TO_MODEL_MAPPING['DUALPQLORA'] = Dual_PQLoraModel
            lora_config.peft_type = 'DUALPQLORA'
        
        elif training_args.mode in ['fedduallastpqfreeze', 'feddualFLpqfreeze', 'feddualMultipqfreeze', 'feddualMultipqfreeze2', ]:
            from models.dual_pqlora_freeze.dual_pqloramodel_freeze import Dual_PQLorafreezeModel
            from peft.peft_model import PEFT_TYPE_TO_MODEL_MAPPING
            PEFT_TYPE_TO_MODEL_MAPPING['DUALPQFreezeLORA'] = Dual_PQLorafreezeModel
            lora_config.peft_type = 'DUALPQFreezeLORA'
        
        elif training_args.mode in ['feddualMultipqfreezeA', 'feddualMulti2pqfreezeA', 'feddualMultipqfreezeA_excludemean', 'feddualMulti2pqfreezeA_excludemean', 'feddualpqfreezeA']:
            from models.dual_pqlora_freezeA.dual_pqloramodel_freezeA import Dual_PQLorafreezeAModel
            from peft.peft_model import PEFT_TYPE_TO_MODEL_MAPPING
            PEFT_TYPE_TO_MODEL_MAPPING['DUALPQFreezeALORA'] = Dual_PQLorafreezeAModel
            lora_config.peft_type = 'DUALPQFreezeALORA'
            
        elif training_args.mode in ['fedduallastpqfullfreeze', 'fedduallastpqfullfreeze_tv',
                                    'feddualMultipqfullfreeze', 'feddualMultipqfullfreeze_tv', 'feddualMultipqfullfreeze_excludemean',
                                    'feddualMulti2pqfullfreeze', 'feddualMulti2pqfullfreeze_tv', 'feddualMulti2pqfullfreeze_excludemean',
                                    'feddualpqfullfreeze','feddualpqfullfreeze_tv',
                                    'feddualMultipqfullfreeze_include', 'feddualMultipqfullfreeze_tv_include', 'feddualMultipqfullfreeze_excludemean_include',
                                    'feddualMultipqfullfreeze256', 'feddualMultipqfullfreeze512','feddualMultipqfullfreeze1024',
                                    'feddualMultipqfullfreeze256_tv', 'feddualMultipqfullfreeze512_tv','feddualMultipqfullfreeze1024_tv',
                                    'feddualMultipqfullfreeze_homoAgg', 'feddualMultipqfullfreeze_excludemean_homoAgg',
                                    'feddualMultipqfullfreeze_distill', 'feddualMultipqfullfreeze_Taskloss', 'feddualMultipqfullfreeze_distillTaskloss'
                                    'feddualBlock2pqfullfreeze', 'feddualBlock4pqfullfreeze'
                                    ]:
            from models.dual_pqlora_freeze_full.dual_pqloramodel_freeze_full import Dual_PQLorafreezeModel
            from peft.peft_model import PEFT_TYPE_TO_MODEL_MAPPING
            PEFT_TYPE_TO_MODEL_MAPPING['DUALPQFullFreezeLORA'] = Dual_PQLorafreezeModel
            lora_config.peft_type = 'DUALPQFullFreezeLORA'
        
        elif training_args.mode in ['feddualMultipqfullfreeze_moe']:
            from models.dual_pqlora_freeze_full_moe.dual_pqloramodel_freeze_full_moe import Dual_PQMOELorafreezeModel
            from peft.peft_model import PEFT_TYPE_TO_MODEL_MAPPING
            PEFT_TYPE_TO_MODEL_MAPPING['DUALPQMOEFullFreezeLORA'] = Dual_PQMOELorafreezeModel
            lora_config.peft_type = 'DUALPQMOEFullFreezeLORA'
        
        elif training_args.mode in ['feddualMultipqfullfreezeA', 'feddualMultipqfullfreezeA_tv', 'feddualMultipqfullfreezeA_excludemean',
                                    'feddualMulti2pqfullfreezeA', 'feddualMulti2pqfullfreezeA_tv', 'feddualMulti2pqfullfreezeA_excludemean',]:
            from models.dual_pqlora_freezeA_full.dual_pqloramodel_freezeA_full import Dual_PQLorafreezeAModel
            from peft.peft_model import PEFT_TYPE_TO_MODEL_MAPPING
            PEFT_TYPE_TO_MODEL_MAPPING['DUALPQFullFreezeALORA'] = Dual_PQLorafreezeAModel
            lora_config.peft_type = 'DUALPQFullFreezeALORA'
            
        elif training_args.mode == 'fedMultipqfullfreeze_ABinit' or training_args.mode == 'fedMulti2pqfullfreeze_ABinit' \
            or training_args.mode == 'fedMultipqfullfreeze256_ABinit' or training_args.mode == 'fedMultipqfullfreeze512_ABinit' or training_args.mode == 'fedMultipqfullfreeze1024_ABinit':
            from models.pqlora_full_init.pqloramodel_full_init import PQLoraModel
            from peft.peft_model import PEFT_TYPE_TO_MODEL_MAPPING
            PEFT_TYPE_TO_MODEL_MAPPING['PQLORAINIT'] = PQLoraModel
            lora_config.peft_type = 'PQLORAINIT'
        # rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)
    
    elif training_args.ia3_enable:
        from peft import IA3Config, get_peft_model
        ia3_config = IA3Config(
            exclude_modules=r".*vision_tower.*",
            target_modules=["k_proj", "v_proj", "down_proj"], 
            feedforward_modules=["down_proj"],
            task_type="CAUSAL_LM",
        )
        
        # create pool
        if training_args.mode in ['fedsim', 'apfl', 'ditto', 'fedours'] or training_args.mode =='feddat':
            from models.dual_ia3.dual_ia3_model import DualIA3Model
            from peft.peft_model import PEFT_TYPE_TO_MODEL_MAPPING
            PEFT_TYPE_TO_MODEL_MAPPING['DUALIA3'] = DualIA3Model
            ia3_config.peft_type = 'DUALIA3'
        
        elif 'L2P' in training_args.mode or 'DAP' in training_args.mode or 'CodaPrompt' in training_args.mode:
            from models.empty_ia3.empty_ia3_model import EmptyIA3Model
            from peft.peft_model import PEFT_TYPE_TO_MODEL_MAPPING
            PEFT_TYPE_TO_MODEL_MAPPING['EMPTYIA3'] = EmptyIA3Model
            ia3_config.peft_type = 'EMPTYIA3'
        
        model = get_peft_model(model, ia3_config)

    if model_args.version in conversation_lib_llava.conv_templates:
        conversation_lib_llava.default_conversation = conversation_lib_llava.conv_templates[model_args.version]
    else:
        conversation_lib_llava.default_conversation = conversation_lib_llava.conv_templates["vicuna_v1"]

    data_args.image_processor = processor.image_processor

    model.config.tokenizer_padding_side = tokenizer.padding_side
    model.config.tokenizer_model_max_length = tokenizer.model_max_length

    # freeze some layers
    # FIXME             
    if training_args.mode in ['sft_only_B_train', 'fedavg_only_B_train', 'fedours_only_B_train', 'fedours_tv_only_B_train']:
        for idx, layer in enumerate(model.base_model.language_model.model.layers):
            for n, p in layer.named_parameters():
                if 'lora_A' in n:
                    print(f"{p} frozen!!")
                    p.requires_grad = False

    elif training_args.mode == 'fedpqfullfreeze' or training_args.mode == 'fedpqfullfreeze_sft':
        from models.pqlora_full.pqloralayer_full import PQLoraFullLayer
        for idx, layer in enumerate(model.base_model.language_model.model.layers):
            for n, p in layer.named_parameters():
                if 'lora_A' in n:
                    p.requires_grad = False
                elif 'lora_B' in n:
                    p.requires_grad = False
                    nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                elif 'lora_P' in n or 'lora_Q' in n:
                    nn.init.zeros_(p)
    
    elif training_args.mode == 'fedpqfullfreezeA' or training_args.mode == 'fedpqfullfreezeA_sft' or training_args.mode == 'fedpqfreezeA' or training_args.mode == 'fedpqfreezeA_sft':
        from models.pqlora_full.pqloralayer_full import PQLoraFullLayer
        for idx, layer in enumerate(model.base_model.language_model.model.layers):
            for n, p in layer.named_parameters():
                if 'lora_A' in n:
                    p.requires_grad = False
                    
    elif training_args.mode == 'fedMultipqfullfreeze' or training_args.mode == 'fedMultipqfullfreeze_sft' or training_args.mode == 'fedMultipqfullfreeze_tv' or training_args.mode == 'fedMultipqfullfreeze_ours':
        from models.pqlora_full.pqloralayer_full import PQLoraFullLayer
        last_layer = len(model.base_model.language_model.model.layers) // 4
        target_layers = [last_layer*1 -1,last_layer*2 -1,last_layer*3 -1,last_layer*4 -1]
        for idx, layer in enumerate(model.base_model.language_model.model.layers):
            if idx in target_layers:
                for n, p in layer.named_parameters():
                    if 'lora_A' in n:
                        p.requires_grad = False
                    elif 'lora_B' in n:
                        p.requires_grad = False
                        nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                    elif 'lora_P' in n or 'lora_Q' in n:
                        nn.init.zeros_(p)
            else:
                for n, m in layer.named_modules():
                    if isinstance(m, PQLoraFullLayer):
                        m.use_pq = False
    
    elif training_args.mode in ['fedBlock2pqfullfreeze', 'fedBlock2pqfullfreeze_sft',
                               'fedBlock4pqfullfreeze', 'fedBlock4pqfullfreeze_sft',]:
        from models.pqlora_full.pqloralayer_full import PQLoraFullLayer
        if 'Block2' in training_args.mode:
            block_layer_num = 2
        elif 'Block4' in training_args.mode:
            block_layer_num = 4
        block_num = len(model.base_model.language_model.model.layers) // block_layer_num
        target_layers = [block_layer_num*(i+1)-1 for i in range(block_num)]
        for idx, layer in enumerate(model.base_model.language_model.model.layers):
            if idx in target_layers:
                for n, p in layer.named_parameters():
                    if 'lora_A' in n:
                        p.requires_grad = False
                    elif 'lora_B' in n:
                        p.requires_grad = False
                        nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                    elif 'lora_P' in n or 'lora_Q' in n:
                        nn.init.zeros_(p)
            else:
                for n, m in layer.named_modules():
                    if isinstance(m, PQLoraFullLayer):
                        m.use_pq = False
    elif training_args.mode in ['feddualBlock2pqfullfreeze', 'feddualBlock4pqfullfreeze']:
        from models.dual_pqlora_freeze_full.dual_pqloralayer_freeze_full import PQLoraFullFreezeLayer
        if 'Block2' in training_args.mode:
            block_layer_num = 2
        elif 'Block4' in training_args.mode:
            block_layer_num = 4
        block_num = len(model.base_model.language_model.model.layers) // block_layer_num
        target_layers = [block_layer_num*(i+1)-1 for i in range(block_num)]
        for idx, layer in enumerate(model.base_model.language_model.model.layers):
            if idx in target_layers:
                for n, p in layer.named_parameters():
                    if 'lora1_A' in n or 'lora2_A' in n:
                        p.requires_grad = False
                    elif 'lora1_B' in n:
                        p.requires_grad = False
                        init_B = torch.empty_like(p)
                        nn.init.kaiming_uniform_(init_B, a=math.sqrt(5))
                        p.data = copy.deepcopy(init_B)
                        new_n = n.replace('lora1_B', 'lora2_B')
                        dict(layer.named_parameters())[new_n].data = copy.deepcopy(init_B)
                        dict(layer.named_parameters())[new_n].requires_grad = False
                    elif 'lora1_P' in n or 'lora2_P' in n or 'lora2_Q' in n or 'lora2_Q' in n:
                        nn.init.zeros_(p)
            else:
                for n, m in layer.named_modules():
                    if isinstance(m, PQLoraFullFreezeLayer):
                        m.use_pq = False
    
    elif training_args.mode in ['fedMultipqfullfreeze256_sft','fedMultipqfullfreeze256',
                                'fedMultipqfullfreeze512_sft','fedMultipqfullfreeze512',
                                'fedMultipqfullfreeze1024_sft','fedMultipqfullfreeze1024',]:
        from models.pqlora_full.pqloralayer_full import PQLoraFullLayer
        last_layer = len(model.base_model.language_model.model.layers) // 4
        target_layers = [last_layer*1 -1,last_layer*2 -1,last_layer*3 -1,last_layer*4 -1]
        if '256' in training_args.mode:
            r = 256
        elif '512' in training_args.mode:
            r = 512
        elif '1024' in training_args.mode:
            r = 1024
        for idx, layer in enumerate(model.base_model.language_model.model.layers):
            if idx in target_layers:
                for n, m in layer.named_modules():
                    if isinstance(m, PQLoraFullLayer):
                        m.r['default'] = r
                        m.lora_alpha['default'] = r*2
                        m.lora_A['default'] = nn.Linear(m.in_features, r, bias=False)
                        m.lora_B['default'] = nn.Linear(r, m.out_features, bias=False)
                        m.lora_P['default'] = nn.Parameter(torch.ones((r,r)))  # Initialized to 1
                        m.lora_Q['default'] = nn.Parameter(torch.zeros((1,r))) # Initialized to 0
                        nn.init.kaiming_uniform_(m.lora_A['default'].weight, a=math.sqrt(5))
                        nn.init.kaiming_uniform_(m.lora_B['default'].weight, a=math.sqrt(5))
                        m.lora_A['default'].weight.requires_grad = False
                        m.lora_B['default'].weight.requires_grad = False
                        
                        nn.init.zeros_(m.lora_P['default'])
                        nn.init.zeros_(m.lora_Q['default'])
                        
            else:
                for n, m in layer.named_modules():
                    if isinstance(m, PQLoraFullLayer):
                        m.use_pq = False
    
    elif training_args.mode in ['feddualMultipqfullfreeze256', 'feddualMultipqfullfreeze512','feddualMultipqfullfreeze1024',
                                'feddualMultipqfullfreeze256_tv', 'feddualMultipqfullfreeze512_tv','feddualMultipqfullfreeze1024_tv']:
        from models.dual_pqlora_freeze_full.dual_pqloralayer_freeze_full import PQLoraFullFreezeLayer
        last_layer = len(model.base_model.language_model.model.layers) // 4
        target_layers = [last_layer*1 -1,last_layer*2 -1,last_layer*3 -1,last_layer*4 -1]
        if '256' in training_args.mode:
            r = 256
        elif '512' in training_args.mode:
            r = 512
        elif '1024' in training_args.mode:
            r = 1024
        for idx, layer in enumerate(model.base_model.language_model.model.layers):
            if idx in target_layers:
                for n, m in layer.named_modules():
                    if isinstance(m, PQLoraFullFreezeLayer):
                        m.r['default'] = r
                        m.lora_alpha['default'] = r*2
                        m.lora1_A['default'] = nn.Linear(m.in_features, r, bias=False)
                        m.lora1_B['default'] = nn.Linear(r, m.out_features, bias=False)
                        m.lora2_A['default'] = copy.deepcopy(m.lora1_A['default'])
                        m.lora2_B['default'] = copy.deepcopy(m.lora1_B['default'])
                        
                        m.lora1_P['default'] = nn.Parameter(torch.ones((r,r)))  # Initialized to 1
                        m.lora1_Q['default'] = nn.Parameter(torch.zeros((1,r))) # Initialized to 0
                        m.lora2_P['default'] = nn.Parameter(torch.ones((r,r)))  # Initialized to 1
                        m.lora2_Q['default'] = nn.Parameter(torch.zeros((1,r))) # Initialized to 0
                        
                        init_A = torch.empty_like(m.lora1_A['default'].weight)
                        nn.init.kaiming_uniform_(init_A, a=math.sqrt(5))
                        # Assign the same values to both lora1 and lora2
                        m.lora1_A['default'].weight.data.copy_(init_A)
                        m.lora2_A['default'].weight.data.copy_(init_A)
                        
                        init_B = torch.empty_like(m.lora1_B['default'].weight)
                        nn.init.kaiming_uniform_(init_B, a=math.sqrt(5))
                        m.lora1_B['default'].weight.data.copy_(init_B)
                        m.lora2_B['default'].weight.data.copy_(init_B)
                        
                        m.lora1_A['default'].weight.requires_grad = False
                        m.lora2_A['default'].weight.requires_grad = False
                        m.lora1_B['default'].weight.requires_grad = False
                        m.lora2_B['default'].weight.requires_grad = False
                        
                        nn.init.zeros_(m.lora1_P['default'])
                        nn.init.zeros_(m.lora2_P['default'])
                        nn.init.zeros_(m.lora1_Q['default'])
                        nn.init.zeros_(m.lora2_Q['default'])
                        
                        m.freeze_AB = True
            else:
                for n, m in layer.named_modules():
                    if isinstance(m, PQLoraFullFreezeLayer):
                        m.use_pq = False
    
    elif training_args.mode in ['fedMultipqfullfreeze256_ABinit','fedMultipqfullfreeze512_ABinit','fedMultipqfullfreeze1024_ABinit']:
        from models.pqlora_full_init.pqloralayer_full_init import PQLoraFullInitLayer
        last_layer = len(model.base_model.language_model.model.layers) // 4
        target_layers = [last_layer*1 -1,last_layer*2 -1,last_layer*3 -1,last_layer*4 -1]
        if '256' in training_args.mode:
            r = 256
        elif '512' in training_args.mode:
            r = 512
        elif '1024' in training_args.mode:
            r = 1024
        for idx, layer in enumerate(model.base_model.language_model.model.layers):
            if idx in target_layers:
                for n, m in layer.named_modules():
                    if isinstance(m,PQLoraFullInitLayer):
                        m.r['default'] = r
                        m.lora_alpha['default'] = r*2
                        m.lora_A['default'] = nn.Linear(m.in_features, r, bias=False)
                        m.lora_B['default'] = nn.Linear(r, m.out_features, bias=False)
                        m.lora_P['default'] = nn.Parameter(torch.ones((r,r)))  # Initialized to 1
                        m.lora_Q['default'] = nn.Parameter(torch.zeros((1,r))) # Initialized to 0
                        m.lora_A['default'].apply(orthonormal_kaiming_uniform_init)
                        m.lora_B['default'].apply(orthonormal_kaiming_uniform_init)
                        m.lora_A['default'].weight.requires_grad = True
                        m.lora_B['default'].weight.requires_grad = False
                        
                        nn.init.zeros_(m.lora_P['default'])
                        nn.init.zeros_(m.lora_Q['default'])
            else:
                for n, m in layer.named_modules():
                    if isinstance(m, PQLoraFullInitLayer):
                        m.use_pq = False
                for n, p in layer.named_parameters():
                    p.requires_grad = False
    
    elif training_args.mode == 'fedMultipqfullfreezeA' or training_args.mode == 'fedMultipqfullfreezeA_sft':
        from models.pqlora_full.pqloralayer_full import PQLoraFullLayer
        last_layer = len(model.base_model.language_model.model.layers) // 4
        target_layers = [last_layer*1 -1,last_layer*2 -1,last_layer*3 -1,last_layer*4 -1]
        for idx, layer in enumerate(model.base_model.language_model.model.layers):
            if idx in target_layers:
                for n, p in layer.named_parameters():
                    if 'lora_A' in n:
                        p.requires_grad = False
                    elif 'lora_P' in n:
                        p.data = torch.eye(p.shape[0]).to(torch.bfloat16)
            else:
                for n, m in layer.named_modules():
                    if isinstance(m, PQLoraFullLayer):
                        m.use_pq = False
    
    elif training_args.mode == 'fedMultipqfreezeA' or training_args.mode == 'fedMultipqfreezeA_sft':
        from models.pqlora.pqloralayer import PQLoraLayer
        last_layer = len(model.base_model.language_model.model.layers) // 4
        target_layers = [last_layer*1 -1,last_layer*2 -1,last_layer*3 -1,last_layer*4 -1]
        for idx, layer in enumerate(model.base_model.language_model.model.layers):
            if idx in target_layers:
                for n, p in layer.named_parameters():
                    if 'lora_A' in n:
                        p.requires_grad = False
                    elif 'lora_P' in n:
                        p.data = torch.ones_like(p)
            else:
                for n, m in layer.named_modules():
                    if isinstance(m, PQLoraLayer):
                        m.use_pq = False
        
    elif training_args.mode == 'fedMultipqfullfreeze_ABinit':
        from models.pqlora_full_init.pqloralayer_full_init import PQLoraFullInitLayer
        last_layer = len(model.base_model.language_model.model.layers) // 4
        target_layers = [last_layer*1 -1,last_layer*2 -1,last_layer*3 -1,last_layer*4 -1]
        for idx, layer in enumerate(model.base_model.language_model.model.layers):
            if idx in target_layers:
                for n, m in layer.named_modules():
                    if 'lora_A.default' in n or 'lora_B.default' in n:
                        m.apply(orthonormal_kaiming_uniform_init)
                for n, p in layer.named_parameters():
                    if 'lora_A' in n:
                        p.requires_grad = True
                    elif 'lora_B' in n:
                        p.requires_grad = False
                    elif 'lora_P' in n or 'lora_Q' in n:
                        nn.init.zeros_(p)
            else:
                for n, m in layer.named_modules():
                    if isinstance(m, PQLoraFullInitLayer):
                        m.use_pq = False
                for n, p in layer.named_parameters():
                    p.requires_grad = False
    
    elif training_args.mode == 'feddualMultipqfreeze':
        from models.dual_pqlora_freeze.dual_pqloralayer_freeze import PQLoraFreezeLayer
        last_layer = len(model.base_model.language_model.model.layers) // 4
        target_layers = [last_layer*1 -1,last_layer*2 -1,last_layer*3 -1,last_layer*4 -1]
        for idx, layer in enumerate(model.base_model.language_model.model.layers):
            if idx in target_layers:
                for n, p in layer.named_parameters():
                    if 'lora1_A' in n or 'lora2_A' in n:
                        p.requires_grad = False
                    elif 'lora1_B' in n:
                        p.requires_grad = False
                        init_B = torch.empty_like(p)
                        nn.init.kaiming_uniform_(init_B, a=math.sqrt(5))
                        p.data = copy.deepcopy(init_B)
                        new_n = n.replace('lora1_B', 'lora2_B')
                        dict(layer.named_parameters())[new_n].data = copy.deepcopy(init_B)
                        dict(layer.named_parameters())[new_n].requires_grad = False
                    elif 'lora1_P' in n or 'lora2_P' in n:
                        nn.init.zeros_(p)
                for n, m in layer.named_modules():
                    if isinstance(m, PQLoraFreezeLayer):
                        m.freeze_AB = True
            else:
                for n, m in layer.named_modules():
                    if isinstance(m, PQLoraFreezeLayer):
                        m.use_pq = False
    
    elif training_args.mode == 'feddualMultipqfreezeA' or training_args.mode == 'feddualMultipqfreezeA_excludemean':
        from models.dual_pqlora_freezeA.dual_pqloralayer_freezeA import PQLoraFreezeALayer
        last_layer = len(model.base_model.language_model.model.layers) // 4
        target_layers = [last_layer*1 -1,last_layer*2 -1,last_layer*3 -1,last_layer*4 -1]
        for idx, layer in enumerate(model.base_model.language_model.model.layers):
            if idx in target_layers:
                for n, p in layer.named_parameters():
                    if 'lora1_A' in n or 'lora2_A' in n:
                        p.requires_grad = False
                    elif 'lora1_P' in n or 'lora2_P' in n:
                        p.data = torch.ones_like(p)
                for n, m in layer.named_modules():
                    if isinstance(m, PQLoraFreezeALayer):
                        m.freeze_AB = True
            else:
                for n, m in layer.named_modules():
                    if isinstance(m, PQLoraFreezeALayer):
                        m.use_pq = False
    elif training_args.mode == 'feddualpqfreezeA':
        from models.dual_pqlora_freezeA.dual_pqloralayer_freezeA import PQLoraFreezeALayer
        for idx, layer in enumerate(model.base_model.language_model.model.layers):
            for n, p in layer.named_parameters():
                if 'lora1_A' in n or 'lora2_A' in n:
                    p.requires_grad = False
                elif 'lora1_P' in n or 'lora2_P' in n:
                    p.data = torch.ones_like(p)
            for n, m in layer.named_modules():
                if isinstance(m, PQLoraFreezeALayer):
                    m.freeze_AB = True

    elif training_args.mode in ['feddualMultipqfullfreeze','feddualMultipqfullfreeze_tv','feddualMultipqfullfreeze_excludemean',
        'feddualMultipqfullfreeze_include','feddualMultipqfullfreeze_tv_include','feddualMultipqfullfreeze_excludemean_include',
        'feddualMultipqfullfreeze_moe','feddualMultipqfullfreeze_homoAgg', 'feddualMultipqfullfreeze_excludemean_homoAgg',]:
        from models.dual_pqlora_freeze_full.dual_pqloralayer_freeze_full import PQLoraFullFreezeLayer
        last_layer = len(model.base_model.language_model.model.layers) // 4
        target_layers = [last_layer*1 -1,last_layer*2 -1,last_layer*3 -1,last_layer*4 -1]
        for idx, layer in enumerate(model.base_model.language_model.model.layers):
            if idx in target_layers:
                for n, p in layer.named_parameters():
                    if 'lora1_A' in n or 'lora2_A' in n:
                        p.requires_grad = False
                    elif 'lora1_B' in n:
                        p.requires_grad = False
                        init_B = torch.empty_like(p)
                        nn.init.kaiming_uniform_(init_B, a=math.sqrt(5))
                        p.data = copy.deepcopy(init_B)
                        new_n = n.replace('lora1_B', 'lora2_B')
                        dict(layer.named_parameters())[new_n].data = copy.deepcopy(init_B)
                        dict(layer.named_parameters())[new_n].requires_grad = False
                    elif 'lora1_P' in n or 'lora2_P' in n or 'lora2_Q' in n or 'lora2_Q' in n:
                        nn.init.zeros_(p)
                for n, m in layer.named_modules():
                    if isinstance(m, PQLoraFullFreezeLayer):
                        m.freeze_AB = True
            else:
                for n, m in layer.named_modules():
                    if isinstance(m, PQLoraFullFreezeLayer):
                        m.use_pq = False

    elif training_args.mode in ['fedMultipqfullfreeze_distill', 'fedMultipqfullfreeze_Taskloss', 'fedMultipqfullfreeze_distillTaskloss']:
        from models.pqlora_full.pqloralayer_full import PQLoraFullLayer
        from models.dual_pqlora_freeze_full.dual_pqloralayer_freeze_full import ProjectMLP
        last_layer = len(model.base_model.language_model.model.layers) // 4
        target_layers = [last_layer*1 -1,last_layer*2 -1,last_layer*3 -1,last_layer*4 -1]
        for idx, layer in enumerate(model.base_model.language_model.model.layers):
            if idx in target_layers:
                for n, p in layer.named_parameters():
                    if 'lora_A' in n:
                        p.requires_grad = False
                    elif 'lora_B' in n:
                        p.requires_grad = False
                        nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                    elif 'lora_P' in n or 'lora_Q' in n:
                        nn.init.zeros_(p)
                if 'Taskloss' in training_args.mode:
                    for n, m in layer.named_modules():
                        if isinstance(m, PQLoraFullLayer) and 'mlp.down_proj' in n:
                            m.lora_F = nn.ModuleDict({})
                            m.lora_F['default'] = ProjectMLP(model.base_model.language_model.config.hidden_size, model.base_model.language_model.config.hidden_size, m.r['default']).to(compute_dtype)
                            m.lora_F['default'].requires_grad = True
                if 'distill' in training_args.mode:
                    for n, m in layer.named_modules():
                        if isinstance(m, PQLoraFullLayer) and 'mlp.down_proj' in n:
                            m.lora_C = nn.ModuleDict({})
                            m.lora_C['default'] = nn.Linear(model.base_model.language_model.config.hidden_size,m.r['default']).to(compute_dtype)
                            m.lora_C['default'].requires_grad = True
            else:
                for n, m in layer.named_modules():
                    if isinstance(m, PQLoraFullLayer):
                        m.use_pq = False
                if 'distill' in training_args.mode:
                    for n, m in layer.named_modules():
                        if isinstance(m, PQLoraFullLayer) and 'mlp.down_proj' in n:
                            m.lora_C = nn.ModuleDict({})
                            m.lora_C['default'] = nn.Linear(model.base_model.language_model.config.hidden_size,m.r['default']).to(compute_dtype)
                            m.lora_C['default'].requires_grad = True

    elif training_args.mode in ['feddualMultipqfullfreeze_distill', 'feddualMultipqfullfreeze_Taskloss', 'feddualMultipqfullfreeze_distillTaskloss']:
        from models.dual_pqlora_freeze_full.dual_pqloralayer_freeze_full import PQLoraFullFreezeLayer, ProjectMLP
        last_layer = len(model.base_model.language_model.model.layers) // 4
        target_layers = [last_layer*1 -1,last_layer*2 -1,last_layer*3 -1,last_layer*4 -1]
        for idx, layer in enumerate(model.base_model.language_model.model.layers):
            if idx in target_layers:
                for n, p in layer.named_parameters():
                    if 'lora1_A' in n or 'lora2_A' in n:
                        p.requires_grad = False
                    elif 'lora1_B' in n:
                        p.requires_grad = False
                        init_B = torch.empty_like(p)
                        nn.init.kaiming_uniform_(init_B, a=math.sqrt(5))
                        p.data = copy.deepcopy(init_B)
                        new_n = n.replace('lora1_B', 'lora2_B')
                        dict(layer.named_parameters())[new_n].data = copy.deepcopy(init_B)
                        dict(layer.named_parameters())[new_n].requires_grad = False
                    elif 'lora1_P' in n or 'lora2_P' in n or 'lora2_Q' in n or 'lora2_Q' in n:
                        nn.init.zeros_(p)
                for n, m in layer.named_modules():
                    if isinstance(m, PQLoraFullFreezeLayer):
                        m.freeze_AB = True
                if 'Taskloss' in training_args.mode:
                    for n, m in layer.named_modules():
                        if isinstance(m, PQLoraFullFreezeLayer) and 'mlp.down_proj' in n:
                            m.lora_F = nn.ModuleDict({})
                            m.lora_F['default'] = ProjectMLP(model.base_model.language_model.config.hidden_size, model.base_model.language_model.config.hidden_size, m.r['default']).to(compute_dtype)
                            m.lora_F['default'].requires_grad = True
                if 'distill' in training_args.mode:
                    for n, m in layer.named_modules():
                        if isinstance(m, PQLoraFullFreezeLayer) and 'mlp.down_proj' in n:
                            m.lora_C = nn.ModuleDict({})
                            m.lora_C['default'] = nn.Linear(model.base_model.language_model.config.hidden_size,m.r['default']).to(compute_dtype)
                            m.lora_C['default'].requires_grad = True
            else:
                for n, m in layer.named_modules():
                    if isinstance(m, PQLoraFullFreezeLayer):
                        m.use_pq = False
                if 'distill' in training_args.mode:
                    for n, m in layer.named_modules():
                        if isinstance(m, PQLoraFullFreezeLayer) and 'mlp.down_proj' in n:
                            m.lora_C = nn.ModuleDict({})
                            m.lora_C['default'] = nn.Linear(model.base_model.language_model.config.hidden_size,m.r['default']).to(compute_dtype)
                            m.lora_C['default'].requires_grad = True
    
    elif training_args.mode == 'feddualMultipqfullfreezeA' or training_args.mode == 'feddualMultipqfullfreezeA_tv' or training_args.mode == 'feddualMultipqfullfreezeA_excludemean':
        from models.dual_pqlora_freezeA_full.dual_pqloralayer_freezeA_full import PQLoraFullFreezeALayer
        last_layer = len(model.base_model.language_model.model.layers) // 4
        target_layers = [last_layer*1 -1,last_layer*2 -1,last_layer*3 -1,last_layer*4 -1]
        for idx, layer in enumerate(model.base_model.language_model.model.layers):
            if idx in target_layers:
                for n, p in layer.named_parameters():
                    if 'lora1_A' in n or 'lora2_A' in n:
                        p.requires_grad = False
                    elif 'lora1_P' in n or 'lora2_P' in n:
                        p.data = torch.eye(p.shape[0]).to(torch.bfloat16)
                for n, m in layer.named_modules():
                    if isinstance(m, PQLoraFullFreezeALayer):
                        m.freeze_AB = True
            else:
                for n, m in layer.named_modules():
                    if isinstance(m, PQLoraFullFreezeALayer):
                        m.use_pq = False
    elif training_args.mode == 'feddualpqfullfreeze' or training_args.mode == 'feddualpqfullfreeze_tv':
        from models.dual_pqlora_freeze_full.dual_pqloralayer_freeze_full import PQLoraFullFreezeLayer
        for idx, layer in enumerate(model.base_model.language_model.model.layers):
            for n, p in layer.named_parameters():
                if 'lora1_A' in n or 'lora2_A' in n:
                    p.requires_grad = False
                elif 'lora1_B' in n:
                        p.requires_grad = False
                        init_B = torch.empty_like(p)
                        nn.init.kaiming_uniform_(init_B, a=math.sqrt(5))
                        p.data = copy.deepcopy(init_B)
                        new_n = n.replace('lora1_B', 'lora2_B')
                        dict(layer.named_parameters())[new_n].data = copy.deepcopy(init_B)
                        dict(layer.named_parameters())[new_n].requires_grad = False
                elif 'lora1_P' in n or 'lora2_P' in n or 'lora2_Q' in n or 'lora2_Q' in n:
                    nn.init.zeros_(p)
            for n, m in layer.named_modules():
                if isinstance(m, PQLoraFullFreezeLayer):
                    m.freeze_AB = True

    model.config.mm_projector_lr = training_args.mm_projector_lr
    
    if hasattr(model.config, "image_token_index"):
        tokenizer.image_token_index = model.config.image_token_index
    
    if training_args.load_pretrained_random or training_args.load_pretrained_pca:
        if training_args.mode in ['fedMultipqfullfreeze_sft', 'fedMultipqfullfreeze', 'fedMultipqfullfreeze_tv', 'fedMultipqfullfreeze_ours', 'fedMultipqfullfreezeA', 'fedMultipqfullfreezeA_sft' 'fedMultipqfreezeA' 'fedMultipqfreezeA_sft',
                                  'fedMultipqfullfreeze256_sft','fedMultipqfullfreeze256',
                                  'fedMultipqfullfreeze512_sft','fedMultipqfullfreeze512',
                                  'fedMultipqfullfreeze1024_sft','fedMultipqfullfreeze1024',
                                  'fedMultipqfullfreeze_distill', 'fedMultipqfullfreeze_Taskloss', 'fedMultipqfullfreeze_distillTaskloss'
                                  ]:
            if training_args.load_pretrained_random:
                if 'llama3.2_1B_vl' in model_args.model_name_or_path:
                    state_dict = torch.load('llava_1b_blockwise_orthnormal_init.pth', map_location='cpu')
                elif 'llama3.2_3B_vl' in model_args.model_name_or_path:
                    state_dict = torch.load('llava_3b_blockwise_orthnormal_init.pth', map_location='cpu')
            elif training_args.load_pretrained_pca:
                if 'llama3.2_1B_vl' in model_args.model_name_or_path:
                    if '256' in training_args.mode:
                        state_dict = torch.load('llava_1b_blockwise_pca_init_r256.pth', map_location='cpu')
                    elif '512' in training_args.mode:
                        state_dict = torch.load('llava_1b_blockwise_pca_init_r512.pth', map_location='cpu')
                    elif '1024' in training_args.mode:
                        state_dict = torch.load('llava_1b_blockwise_pca_init_r1024.pth', map_location='cpu')
                    else:
                        state_dict = torch.load('llava_1b_blockwise_pca_init.pth', map_location='cpu')
                elif 'llama3.2_3B_vl' in model_args.model_name_or_path:
                    if '256' in training_args.mode:
                        state_dict = torch.load('llava_3b_blockwise_pca_init_r256.pth', map_location='cpu')
                    elif '512' in training_args.mode:
                        state_dict = torch.load('llava_3b_blockwise_pca_init_r512.pth', map_location='cpu')
                    elif '1024' in training_args.mode:
                        state_dict = torch.load('llava_3b_blockwise_pca_init_r1024.pth', map_location='cpu')
                    else:
                        state_dict = torch.load('llava_3b_blockwise_pca_init.pth', map_location='cpu')
            if 'freezeA' in training_args.mode:
                new_state_dict = {}
                for k, v in state_dict.items():
                    if 'lora_P' in k:
                        if 'full' in training_args.mode:
                            v.data = torch.eye(v.shape[0]).to(torch.bfloat16)
                        else:
                            v = torch.ones(v.shape[0]).to(torch.bfloat16)
                    elif 'lora_B' in k:
                        v.data = torch.zeros_like(v)
                    elif 'lora_Q' in k:
                        nn.init.kaiming_uniform_(v, a=math.sqrt(5))
                    new_state_dict[k] = v
                state_dict = new_state_dict
            
            model.load_state_dict(state_dict, strict=False)
        elif training_args.mode in ['feddualMultipqfullfreeze', 'feddualMultipqfullfreeze_tv', 'feddualMultipqfullfreeze_excludemean',
                                    'feddualMultipqfullfreezeA', 'feddualMultipqfullfreezeA_tv', 'feddualMultipqfullfreezeA_excludemean',
                                    'feddualMultipqfreezeA', 'feddualMultipqfreezeA_excludemean',
                                    'feddualMultipqfullfreeze_include', 'feddualMultipqfullfreeze_tv_include', 'feddualMultipqfullfreeze_excludemean_include','feddualMultipqfullfreeze_moe',
                                    'feddualMultipqfullfreeze256','feddualMultipqfullfreeze512','feddualMultipqfullfreeze1024',
                                    'feddualMultipqfullfreeze256_tv','feddualMultipqfullfreeze512_tv','feddualMultipqfullfreeze1024_tv'
                                    'feddualMultipqfullfreeze_homoAgg', 'feddualMultipqfullfreeze_excludemean_homoAgg',
                                    'feddualMultipqfullfreeze_distill', 'feddualMultipqfullfreeze_Taskloss', 'feddualMultipqfullfreeze_distillTaskloss'
                                    ]:
            if training_args.load_pretrained_random:
                if 'llama3.2_1B_vl' in model_args.model_name_or_path:
                    state_dict = torch.load('llava_1b_blockwise_orthnormal_init.pth', map_location='cpu')
                elif 'llama3.2_3B_vl' in model_args.model_name_or_path:
                    state_dict = torch.load('llava_3b_blockwise_orthnormal_init.pth', map_location='cpu')
            elif training_args.load_pretrained_pca:
                if 'llama3.2_1B_vl' in model_args.model_name_or_path:
                    if '256' in training_args.mode:
                        state_dict = torch.load('llava_1b_blockwise_pca_init_r256.pth', map_location='cpu')
                    elif '512' in training_args.mode:
                        state_dict = torch.load('llava_1b_blockwise_pca_init_r512.pth', map_location='cpu')
                    elif '1024' in training_args.mode:
                        state_dict = torch.load('llava_1b_blockwise_pca_init_r1024.pth', map_location='cpu')
                    else:
                        state_dict = torch.load('llava_1b_blockwise_pca_init.pth', map_location='cpu')
                elif 'llama3.2_3B_vl' in model_args.model_name_or_path:
                    if '256' in training_args.mode:
                        state_dict = torch.load('llava_3b_blockwise_pca_init_r256.pth', map_location='cpu')
                    elif '512' in training_args.mode:
                        state_dict = torch.load('llava_3b_blockwise_pca_init_r512.pth', map_location='cpu')
                    elif '1024' in training_args.mode:
                        state_dict = torch.load('llava_3b_blockwise_pca_init_r1024.pth', map_location='cpu')
                    else:
                        state_dict = torch.load('llava_3b_blockwise_pca_init.pth', map_location='cpu')
            new_state_dict = {}
            for k, v in state_dict.items():
                new_k1 = k.replace('lora', 'lora1')
                new_k2 = k.replace('lora', 'lora2')
                if 'freezeA' in training_args.mode and 'lora_P' in k:
                    if 'full' in training_args.mode:
                        v.data = torch.eye(v.shape[0]).to(torch.bfloat16)
                    else:
                        v = torch.ones(v.shape[0]).to(torch.bfloat16)
                elif 'freezeA' in training_args.mode and 'lora_B' in k:
                    v.data = torch.zeros_like(v)
                elif 'freezeA' in training_args.mode and 'lora_Q' in k:
                    nn.init.kaiming_uniform_(v, a=math.sqrt(5))
                new_state_dict[new_k1] = v
                new_state_dict[new_k2] = v
            model.load_state_dict(new_state_dict, strict=False)
    
        print('load pretrained lora')
    
    total_count = 0
    for n, p in model.named_parameters():
        if p.requires_grad:
            # print(n, p.shape)
            p.data = p.data.to(compute_dtype) 
            total_count += p.numel()
    print(f"trainable param num: {total_count}")
    return model, tokenizer, processor, data_args

def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer_pt_utils import get_parameter_names

def get_decay_parameter_names(model):
    """
    Get all parameter names that weight decay will be applied to

    Note that some models implement their own layernorm instead of calling nn.LayerNorm, weight decay could still
    apply to those modules since this function only filter out instance of nn.LayerNorm
    """
    decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    return decay_parameters



# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        # Only save Adapter
        keys_to_match = ['mm_projector']
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(['embed_tokens', 'embed_in'])

        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa
        

def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param

from torch import nn

def load_deepspeed(state_dict, module: nn.Module, prefix="", strict=True):
    import deepspeed
    # because zero3 puts placeholders in model params, this context
    # manager gathers (unpartitions) the params of the current layer, then loads from
    # the state dict and then re-partitions them again
    with deepspeed.zero.GatheredParameters(list(module.parameters(recurse=False)), modifier_rank=0):
        if deepspeed.comm.get_rank() == 0:
            module._load_from_state_dict(state_dict, prefix, {}, strict, [], [], [])
            # module.load_state_dict(state_dict, strict=strict)

    for name, child in module._modules.items():
        if child is not None:
            load_deepspeed(state_dict, child, prefix + name + ".")

import numpy as np
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference

def configure_online_datastream(sub_dataset, num_iterations, training_args, client_id, memory, memory_count, memory_size, total_batchsize):
    iteration = 0
    datalist = []
    iter_ratio = num_iterations / len(sub_dataset)
    T = 0.125
    count_decay_ratio = 0.99
    
    if not training_args.is_streamonly:
        # memory-only
        # for i, sample in enumerate(sub_dataset):
        #     if len(memory[client_id]) == memory_size:
        #         memory[client_id].pop(random.randrange(memory_size))
        #     memory[client_id].append(sample)
        #     iteration += iter_ratio
        #     if iteration >= 1:
        #         for _ in range(int(iteration)):
        #             batch = random.sample(memory[client_id], k=min(len(memory[client_id]), total_batchsize))
        #             mul = (total_batchsize//len(batch)) + 1
        #             batch = (batch*mul)[:total_batchsize]
        #             datalist.extend(batch[:])
        #             iteration -= 1
        # if len(datalist) < num_iterations*total_batchsize:
            # batch = random.sample(memory[client_id], k=min(len(memory[client_id]), total_batchsize))
            # mul = (total_batchsize//len(batch)) + 1
            # batch = (batch*mul)[:total_batchsize]
            # datalist.extend(batch[:])
        
        # memory only: priority-based sampling
        for i, sample in enumerate(sub_dataset):
            if len(memory[client_id]) == memory_size:
                pop_index = random.randrange(memory_size)
                memory[client_id].pop(pop_index)
                memory_count[client_id] = np.delete(memory_count[client_id], pop_index, 0)
            
            memory[client_id].append(sample)
            memory_count[client_id] = np.append(memory_count[client_id], 0)
            iteration += iter_ratio
            if iteration >= 1:
                for _ in range(int(iteration)):
                    # if len(memory[client_id]) > total_batchsize:
                        # count_decay_ratio = total_batchsize / (len(memory[client_id])*k_coeff)
                    memory_count[client_id] *= count_decay_ratio
                    sample_score = memory_count[client_id]
                    weight = softmax(-sample_score/T)
                    sample_idx = np.random.choice(len(memory[client_id]), min(len(memory[client_id]), total_batchsize), p=weight, replace=False)
                    batch = [memory[client_id][idx] for idx in sample_idx]
                    mul = (total_batchsize//len(batch)) + 1
                    batch = (batch*mul)[:total_batchsize]
                    datalist.extend(batch[:])
                    iteration -= 1
                    for idx in sample_idx:
                        memory_count[client_id][idx] += 1
        if len(datalist) < num_iterations*total_batchsize:
            memory_count[client_id] *= count_decay_ratio
            sample_score = memory_count[client_id]
            weight = softmax(-sample_score/T)
            sample_idx = np.random.choice(len(memory[client_id]), min(len(memory[client_id]), total_batchsize), p=weight, replace=False)
            batch = [memory[client_id][idx] for idx in sample_idx]
            mul = (total_batchsize//len(batch)) + 1
            batch = (batch*mul)[:total_batchsize]
            datalist.extend(batch[:])
            for idx in sample_idx:
                memory_count[client_id][idx] += 1
    else:
        # stream-only
        datalist = sub_dataset[:num_iterations*total_batchsize]
    
    return datalist

def get_keys_to_del(training_args, new_global_state_dict):
    keys_to_del = []
    if training_args.mode in ['fedours', 'fedours_tv', 'fedours_excludemean', 'fedours_include', 'fedours_tv_include', 'fedours_excludemean_include', 'fedours_excludemean_hetero',
                              'fedours_moe', 'fedours_only_B_train', 'fedours_tv_only_B_train', 'fedours_hetero', 'feddualMultipqfullfreeze_homoAgg', 'feddualMultipqfullfreeze_excludemean_homoAgg']:
        for k in new_global_state_dict.keys():
            if 'lora2' in k or 'ia3_l_2' in k or 'ia3_generator_2' in k or 'lang_prompt_ia3_pool_2' in k \
            or 'lang_prompt_dap_key_embeddings_2' in k or 'lang_prompt_downsample_2' in k or 'lang_prompt_norm_2' in k \
            or 'lang_prompt_downsample_kv_2' in k or 'lang_prompt_downsample_mlp_2' in k \
            or 'lora_w_gate' in k or 'lora_w_noise' in k:
                keys_to_del.append(k)
    elif training_args.mode == 'fedpq' or training_args.mode == 'fedpq_sft':
        for k in new_global_state_dict.keys():
            if 'lora_P' not in k and 'lora_Q' not in k:
                keys_to_del.append(k)
    elif training_args.mode == 'fedlastpq' or training_args.mode == 'fedlastpqfreeze' or training_args.mode == 'fedlastpqfullfreeze' or training_args.mode == 'fedlastpqfullfreeze_sft' or training_args.mode == 'fedlastpqfullfreeze_tv'  or training_args.mode == 'fedlastpqfullfreeze_ours':
        layer_num = []
        for k in new_global_state_dict.keys():
            if 'layers.' in k:
                layer_num.append(int(k.split('.')[5]))
        layer_num = sorted(list(set(layer_num)))
        
        layers_to_del = layer_num[:-1]
        for k in new_global_state_dict.keys():
            if 'layers.' in k and int(k.split('.')[5]) in layers_to_del or ('lora_P' not in k and 'lora_Q' not in k):
                keys_to_del.append(k)
    
    elif training_args.mode in ['fedBlock2pqfullfreeze', 'fedBlock2pqfullfreeze_sft',
                               'fedBlock4pqfullfreeze', 'fedBlock4pqfullfreeze_sft',]:
        layer_num = []
        for k in new_global_state_dict.keys():
            if 'layers.' in k:
                layer_num.append(int(k.split('.')[5]))
        layer_num = sorted(list(set(layer_num)))
        
        if 'Block2' in training_args.mode:
            block_layer_num = 2
        elif 'Block4' in training_args.mode:
            block_layer_num = 4
        
        block_num = len(layer_num) // block_layer_num
        
        for i in range(block_num, 0, -1):
            del layer_num[block_layer_num*i - 1]
        
        layers_to_del = layer_num
        for k in new_global_state_dict.keys():
            if 'layers.' in k and int(k.split('.')[5]) in layers_to_del or ('lora_P' not in k and 'lora_Q' not in k):
                keys_to_del.append(k)
    elif training_args.mode in ['feddualBlock2pqfullfreeze', 'feddualBlock4pqfullfreeze']:
        layer_num = []
        for k in new_global_state_dict.keys():
            if 'layers.' in k:
                layer_num.append(int(k.split('.')[5]))
        layer_num = sorted(list(set(layer_num)))
        
        if 'Block2' in training_args.mode:
            block_layer_num = 2
        elif 'Block4' in training_args.mode:
            block_layer_num = 4
        
        block_num = len(layer_num) // block_layer_num
        
        for i in range(block_num, 0, -1):
            del layer_num[block_layer_num*i - 1]
                    
        layers_to_del = layer_num
        for k in new_global_state_dict.keys():
            if 'layers.' in k and int(k.split('.')[5]) in layers_to_del or ('lora1_P' not in k and 'lora1_Q' not in k):
                keys_to_del.append(k)
    elif training_args.mode in ['fedMultipqfullfreeze','fedMultipqfullfreeze_sft','fedMultipqfullfreeze_tv','fedMultipqfullfreeze_ours',
                                'fedMultipqfullfreezeA','fedMultipqfullfreezeA_sft','fedMultipqfreezeA','fedMultipqfreezeA_sft',
                                'fedMultipqfullfreeze256', 'fedMultipqfullfreeze256_sft',
                                'fedMultipqfullfreeze512', 'fedMultipqfullfreeze512_sft',
                                'fedMultipqfullfreeze1024', 'fedMultipqfullfreeze1024_sft',
                                'fedMultipqfullfreeze_distill', 'fedMultipqfullfreeze_Taskloss', 'fedMultipqfullfreeze_distillTaskloss']:
        layer_num = []
        for k in new_global_state_dict.keys():
            if 'layers.' in k:
                layer_num.append(int(k.split('.')[5]))
        layer_num = sorted(list(set(layer_num)))
        
        index = len(layer_num) // 4
        del layer_num[index*4-1]
        del layer_num[index*3-1]
        del layer_num[index*2-1]
        del layer_num[index*1-1]
        
        layers_to_del = layer_num
        for k in new_global_state_dict.keys():
            if 'layers.' in k and int(k.split('.')[5]) in layers_to_del or ('lora_P' not in k and 'lora_Q' not in k):
                keys_to_del.append(k)
    elif training_args.mode == 'fedMulti2pqfullfreeze' or training_args.mode == 'fedMulti2pqfullfreeze_sft' or training_args.mode == 'fedMulti2pqfullfreeze_tv' or training_args.mode == 'fedMulti2pqfullfreeze_ours' or training_args.mode == 'fedMulti2pqfullfreezeA' or training_args.mode == 'fedMulti2pqfreezeA':
        layer_num = []
        for k in new_global_state_dict.keys():
            if 'layers.' in k:
                layer_num.append(int(k.split('.')[5]))
        layer_num = sorted(list(set(layer_num)))
        
        index = len(layer_num) // 4
        del layer_num[index*4-1]
        del layer_num[index*4-2]
        del layer_num[index*3-1]
        del layer_num[index*3-2]
        del layer_num[index*2-1]
        del layer_num[index*2-2]
        del layer_num[index*1-1]
        del layer_num[index*1-2]
        
        layers_to_del = layer_num
        for k in new_global_state_dict.keys():
            if 'layers.' in k and int(k.split('.')[5]) in layers_to_del or ('lora_P' not in k and 'lora_Q' not in k):
                keys_to_del.append(k)

    elif training_args.mode == 'fedFLpq' or training_args.mode == 'fedFLpqfreeze':
        layer_num = []
        for k in new_global_state_dict.keys():
            if 'layers.' in k:
                layer_num.append(int(k.split('.')[5]))
        layer_num = sorted(list(set(layer_num)))
        
        layers_to_del = layer_num[1:-1]
        for k in new_global_state_dict.keys():
            if 'layers.' in k and int(k.split('.')[5]) in layers_to_del or ('lora_P' not in k and 'lora_Q' not in k):
                keys_to_del.append(k)
    elif training_args.mode == 'fedFMLpq':
        layer_num = []
        for k in new_global_state_dict.keys():
            if 'layers.' in k:
                layer_num.append(int(k.split('.')[5]))
        layer_num = sorted(list(set(layer_num)))
        
        mid_layer = int(len(layer_num)/2) - 1
        del layer_num[mid_layer]
        layers_to_del = layer_num[1:-1]
        for k in new_global_state_dict.keys():
            if 'layers.' in k and int(k.split('.')[5]) in layers_to_del or ('lora_P' not in k and 'lora_Q' not in k):
                keys_to_del.append(k)
    elif training_args.mode == 'fedpqfullfreeze' or training_args.mode == 'fedpqfullfreeze_sft' or training_args.mode == 'fedpqfullfreezeA_sft' or training_args.mode == 'fedpqfullfreezeA' or training_args.mode == 'fedpqfreezeA_sft' or training_args.mode == 'fedpqfreezeA':
        for k in new_global_state_dict.keys():
            if 'lora_P' not in k and 'lora_Q' not in k:
                keys_to_del.append(k)
    elif training_args.mode == 'feddualpq' or training_args.mode == 'feddualpqfullfreeze' or training_args.mode == 'feddualpqfullfreeze_tv' or training_args.mode == 'feddualpqfreezeA':
        for k in new_global_state_dict.keys():
            if 'lora1_P' not in k and 'lora1_Q' not in k:
                keys_to_del.append(k)
    elif training_args.mode == 'fedduallastpq' or training_args.mode == 'fedduallastpqfreeze' or training_args.mode == 'fedduallastpqfullfreeze' or training_args.mode == 'fedduallastpqfullfreeze_tv':
        layer_num = []
        for k in new_global_state_dict.keys():
            if 'layers.' in k:
                layer_num.append(int(k.split('.')[5]))
        layer_num = sorted(list(set(layer_num)))
        
        layers_to_del = layer_num[:-1]
        for k in new_global_state_dict.keys():
            if 'layers.' in k and int(k.split('.')[5]) in layers_to_del or ('lora1_P' not in k and 'lora1_Q' not in k):
                keys_to_del.append(k)
    elif training_args.mode == 'feddualFLpq' or training_args.mode == 'feddualFLpqfreeze':
        layer_num = []
        for k in new_global_state_dict.keys():
            if 'layers.' in k:
                layer_num.append(int(k.split('.')[5]))
        layer_num = sorted(list(set(layer_num)))
        
        layers_to_del = layer_num[1:-1]
        for k in new_global_state_dict.keys():
            if 'layers.' in k and int(k.split('.')[5]) in layers_to_del or ('lora1_P' not in k and 'lora1_Q' not in k):
                keys_to_del.append(k)
    elif training_args.mode == 'feddualFMLpq':
        layer_num = []
        for k in new_global_state_dict.keys():
            if 'layers.' in k:
                layer_num.append(int(k.split('.')[5]))
        layer_num = sorted(list(set(layer_num)))
        
        mid_layer = int(len(layer_num)/2) - 1
        del layer_num[mid_layer]
        
        layers_to_del = layer_num[1:-1]
        for k in new_global_state_dict.keys():
            if 'layers.' in k and int(k.split('.')[5]) in layers_to_del or ('lora1_P' not in k and 'lora1_Q' not in k):
                keys_to_del.append(k)
    elif training_args.mode in ['feddualMultipqfreeze','feddualMultipqfullfreeze','feddualMultipqfullfreeze_tv','feddualMultipqfullfreeze_excludemean',
                                'feddualMultipqfullfreezeA','feddualMultipqfullfreezeA_tv','feddualMultipqfullfreezeA_excludemean',
                                'feddualMultipqfreezeA','feddualMultipqfreezeA_excludemean','feddualMultipqfullfreeze_moe',
                                'feddualMultipqfullfreeze_include','feddualMultipqfullfreeze_tv_include','feddualMultipqfullfreeze_excludemean_include',
                                'feddualMultipqfullfreeze256','feddualMultipqfullfreeze512','feddualMultipqfullfreeze1024',
                                'feddualMultipqfullfreeze256_tv','feddualMultipqfullfreeze512_tv','feddualMultipqfullfreeze1024_tv',
                                'feddualMultipqfullfreeze_distill', 'feddualMultipqfullfreeze_Taskloss', 'feddualMultipqfullfreeze_distillTaskloss',
                                ]:
        layer_num = []
        for k in new_global_state_dict.keys():
            if 'layers.' in k:
                layer_num.append(int(k.split('.')[5]))
        layer_num = sorted(list(set(layer_num)))
        
        index = len(layer_num) // 4
        del layer_num[index*4-1]
        del layer_num[index*3-1]
        del layer_num[index*2-1]
        del layer_num[index*1-1]
        
        layers_to_del = layer_num
        for k in new_global_state_dict.keys():
            if 'layers.' in k and int(k.split('.')[5]) in layers_to_del or ('lora1_P' not in k and 'lora1_Q' not in k):
                keys_to_del.append(k)
    elif training_args.mode == 'feddualMulti2pqfullfreeze' or training_args.mode == 'feddualMulti2pqfullfreeze_tv'  or training_args.mode == 'feddualMulti2pqfullfreeze_excludemean' \
        or training_args.mode == 'feddualMulti2pqfullfreezeA' or training_args.mode == 'feddualMulti2pqfullfreezeA_tv' or training_args.mode == 'feddualMulti2pqfullfreezeA_excludemean' \
        or training_args.mode == 'feddualMulti2pqfreezeA' or training_args.mode == 'feddualMulti2pqfreezeA_excludemean':
        layer_num = []
        for k in new_global_state_dict.keys():
            if 'layers.' in k:
                layer_num.append(int(k.split('.')[5]))
        layer_num = sorted(list(set(layer_num)))
        
        index = len(layer_num) // 4
        del layer_num[index*4-1]
        del layer_num[index*4-2]
        del layer_num[index*3-1]
        del layer_num[index*3-2]
        del layer_num[index*2-1]
        del layer_num[index*2-2]
        del layer_num[index*1-1]
        del layer_num[index*1-2]
        
        layers_to_del = layer_num
        for k in new_global_state_dict.keys():
            if 'layers.' in k and int(k.split('.')[5]) in layers_to_del or ('lora1_P' not in k and 'lora1_Q' not in k):
                keys_to_del.append(k)

    return keys_to_del

def orthonormal_kaiming_uniform_init(m):
    if isinstance(m, nn.Linear):
        # Get the current shape of the weight matrix
        rows, cols = m.weight.size()

        # Initialize the weight matrix with Kaiming Uniform
        a = nn.init.kaiming_uniform_(torch.empty(rows, cols).cuda(), a=math.sqrt(5))

        # Ensure the matrix is contiguous
        a = a.contiguous()

        # Perform Singular Value Decomposition
        u, _, v = torch.svd(a, some=False)
        
        u = u.contiguous()
        v = v.contiguous()

        # Use U or V from SVD based on the shape of the weight matrix
        if rows > cols:
            m.weight.data = u[:, :cols].to(torch.bfloat16)
        else:
            m.weight.data = v[:rows, :].to(torch.bfloat16)

        # Optional: Initialize biases to zero
        if m.bias is not None:
            m.bias.data.zero_()

import random
from federated_methods.fedours import fedours_ema_distill_create_trainer
def get_task_vectors(model, tokenizer, processor, train_datalists, training_args, data_args, global_state_dict_list, make_supervised_data_module, model2):
    random.seed(training_args.seed)
    client_task_vectors = []
    for client_id in range(len(train_datalists)):
        datalist = train_datalists[client_id][0]['datalist']
        
        sub_datalist = random.sample(datalist, 4*20)
        
        data_module = make_supervised_data_module(client_data=sub_datalist, # sub_dataset
                                                tokenizer=tokenizer,
                                                processor=processor,
                                                data_args=copy.deepcopy(data_args))
    
        extra_state_dict_dict = {}
        extra_state_dict_dict['client_id']=0
        extra_state_dict_dict['curr_round']=0
        extra_state_dict_dict['fisher_freq'] = 1
        extra_state_dict_dict['test_datalist'] = []
        extra_state_dict_dict['processor'] = processor
        extra_state_dict_dict['data_args'] = data_args
        extra_state_dict_dict['model2'] = model2
        copy_training_args = copy.deepcopy(training_args)
        copy_training_args.per_gpu_train_batch_size = 4
        copy_training_args.gradient_accumulation_steps = 1
        trainer = fedours_ema_distill_create_trainer(model, tokenizer, copy_training_args, data_module, extra_state_dict_dict)

        results = trainer.train()
        
        task_vector = trainer.task_vector
        
        client_task_vectors.append(task_vector)
        
        trainer.deepspeed.empty_partition_cache()
        del trainer
        
        with torch.no_grad():
            model.load_state_dict(global_state_dict_list[client_id], strict=False)
    
    extra_state_dict_dict['fisher_freq']=5
    return client_task_vectors
