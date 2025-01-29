# PFCL

## Environment setting
```bash
conda create -n fcl python=3.10
conda activate fcl
pip install transformers==4.47.1
pip install torch==2.3.0 torchvision --index-url https://download.pytorch.org/whl/cu118
pip install flash-attn==2.5.8 --no-build-isolation
pip install peft bitsandbytes pandas kornia opencv-python timm torch_optimizer easydict pycocoevalcap sentencepiece protobuf trl==0.8.6 deepspeed==0.14.0 loguru captum POT jsonlines numpy==1.26.4 accelerate==0.29.3
pip install -U scikit-learn
```

```bash
gdrive files download 1aHRzTuAjthxQoV3k0myevHzBCsLekclY
gdrive files download 16_tCIWdUNPS7VCNcFux4tgLBR_2O9tcn
```