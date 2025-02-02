# PFCL

## Environment setting
```bash
conda create -n fcl python=3.10
conda activate fcl
pip install transformers==4.47.1
pip install torch==2.3.0 torchvision --index-url https://download.pytorch.org/whl/cu118
pip install flash-attn==2.5.8 --no-build-isolation
pip install peft==0.14.0 bitsandbytes pandas kornia opencv-python timm torch_optimizer easydict pycocoevalcap sentencepiece protobuf trl==0.8.6 deepspeed==0.15.2 loguru captum POT jsonlines numpy==1.26.4 accelerate==0.29.3
pip install -U scikit-learn
```

- init weight for 1B, 3B
```bash
gdrive files download 1b90paEEbwvv608S7KW-XcXKcCcXWAyhS
gdrive files download 1gUh9JvVpJJqOXW0mzoWKiAF_0OuGFcKW
```

## Dataset preprocessing Again

```bash
python ./preprocess_again/preprocess_again.py
python ./preprocess_again/preprocess_aqua_again.py
python ./preprocess_again/preprocess_bongard_hoi_again.py
python ./preprocess_again/preprocess_bongard_hoi_query_again.py
python ./preprocess_again/preprocess_coinstruct_db_again.py
python ./preprocess_again/preprocess_fashioniq_again.py
python ./preprocess_again/preprocess_patterncom_again.py
```