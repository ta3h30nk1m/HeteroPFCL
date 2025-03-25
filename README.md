# PFCL

## Environment setting
```bash
conda create -n fcl2 python=3.10
conda activate fcl2
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
    - blockwise (4 layers) - PCA
```bash
gdrive files download 1YxdZKJosR53oyi_kMiMb8Fu9WldIyUT5
gdrive files download 1iQYOO5VNRTA282JZCsNlSWLF4x4n6R5c
```
    - blockwise (2 layers) - PCA
```bash
gdrive files download 194GaXFFlz3-T7rQQpnzQI1sp6FgIKA0D
gdrive files download 1wnh1gelZrf810q2hSMHg0K_j_FZKp1jU
```

    - blockwise Optimal (2 layers) - PCA
```bash
gdrive files download 1LG81Bsz24FnQQnD7YFA_2GGfg03syQYg
gdrive files download 1xa_WluV_fe0RjW0bGndM97Gdk_QB9Igk
```
    - blockwise Optimal (4 layers) - PCA
```bash
gdrive files download 1H7er2B6W3uLQUCkDlncFuED-opuNpEhy
gdrive files download 1py9w8dDAQ-iG7m8x-s1Woj3l3h8_MvXu
```
    - blockwise Optimal (8 layers) - PCA
```bash
gdrive files download 1RdwczYGzEmQetTizqo6s63OAGLP8yJj8
gdrive files download 1rjqa8tLrrCx3K5p167Z8EruqYEaJ7P7H
```

    - blockwise - PCA (R256)
```bash
gdrive files download 1IS4fEK0-KmVJ9rgmCtIzhYEDjpJWSW8x
gdrive files download 1slnzeTumli3s5nTWkpLzJfRq0KW8-cUt
```
    - blockwise - PCA (R512)
```bash
gdrive files download 1tHjBCnYtObyQibuqdSrk7vvCEBRJemkA
gdrive files download 1IZLEG2F1F2YiR-5QhwqldLnJvhup45Ao
```
    - blockwise - PCA (R1024)
```bash
gdrive files download 1uDJ3eMnsbdHcfxqphpPL57ovk1HqKEOU
gdrive files download 1ADtJlXtyMWNVDk_W9GWG5CLawj64tiQ6
```

    - llama - blockwise - PCA (R16)
```bash
gdrive files download 1lYofrpJhLfFdg4-ayZnQ73-qIZVSdBxk
gdrive files download 1sygf0_hkyMYxMCbeX6RUmSjbfAZ3SZ3I
```

## New data instruction

1. In `dataset` folder, run the following script files:
```bash
bash Fashion.sh
bash HRVQA.sh
bash Pair_TF.sh
bash KGQA.sh
bash Bongard.sh
bash iconqa.sh
bash CoInstruct.sh
bash Visual_Relation.sh
bash Visual_storytelling.sh
bash MultiVQA_small.sh
```

2. Run the following preprocessing python codes:
- Fashion
```bash
python ./preprocess/preprocess_Fashion200K.py
```
- HRVQA
```bash
cd ./dataset/HRVQA/jsons
python task_split.py
cd ../../..
python ./preprocess/preprocess_HRVQA.py
```
- Pair_TF
```bash
python ./preprocess/preprocess_NLVR2.py
python ./preprocess/preprocess_PatternCom.py
```

- KGQA
```bash
python ./preprocess/preprocess_WebQA.py
python ./preprocess/preprocess_TQA.py
python ./preprocess/preprocess_AQUA.py
```

- Bongard
```bash
python ./preprocess/preprocess_Bongard2.py
python ./preprocess/preprocess_Bongard_query.py
python ./preprocess/preprocess_Bongard_HOI.py
python ./preprocess/preprocess_Bongard_HOI_query.py
```

- IconQA
```bash
python ./preprocess/preprocess_iconqa.py
```

- CoInstruct
```bash
python ./preprocess/preprocess_coinstruct.py
```

- Visual_Relation
```bash
python ./preprocess/preprocess_SpotDiff.py
python ./preprocess/preprocess_Bird2Words.py
python ./preprocess/preprocess_CLEVR.py
python ./preprocess/preprocess_IEdit.py
```

- Visual_storytelling
```bash
python ./preprocess/preprocess_PororoSV.py
python ./preprocess/preprocess_FlintstonesSV.py
python ./preprocess/preprocess_VIST.py
python ./preprocess/preprocess_AESOP.py
```

- MultiVQA_small
```bash
python ./preprocess/preprocess_VISION.py
python ./preprocess/preprocess_VizWiz.py
python ./preprocess/preprocess_MIT.py
```

## Dataset preprocessing Again

```bash
python ./preprocess_again2/preprocess_again.py
python ./preprocess_again2/preprocess_aqua_again.py
python ./preprocess_again2/preprocess_bongard_hoi_again.py
python ./preprocess_again2/preprocess_bongard_hoi_query_again.py
python ./preprocess_again2/preprocess_coinstruct_db_again.py
python ./preprocess_again2/preprocess_fashioniq_again.py
python ./preprocess_again2/preprocess_patterncom_again.py
```