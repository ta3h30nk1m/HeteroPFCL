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
    - blockwise (4 layers)
```bash
gdrive files download 1b90paEEbwvv608S7KW-XcXKcCcXWAyhS
gdrive files download 1gUh9JvVpJJqOXW0mzoWKiAF_0OuGFcKW
```
    - blockwise2 (8 layers)
```bash
gdrive files downlaod 1daRFCws40l7dKstM6b_takNBF0xShZm7
gdrive files download 1xqGdFk49_3J7Y2Wtyn9eSKv9FL8lqSDe
```
    - blockwise (4 layers) - PCA
```bash
gdrive files download 1YxdZKJosR53oyi_kMiMb8Fu9WldIyUT5
gdrive files download 1iQYOO5VNRTA282JZCsNlSWLF4x4n6R5c
```
    - blockwise2 (8 layers) - PCA
```bash
gdrive files downlaod 1OTDkcu1hf_U09LM6ddyySFfJrAIZssfw
gdrive files download 1u53165H0il1leqA4dyo2k7MFNjd1u3_w
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