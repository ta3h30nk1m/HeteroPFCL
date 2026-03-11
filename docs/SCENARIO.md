# Federated Learning Scenarios



- old scenario (기존 실험에 사용했던 시나리오)
    - sc132: DRAKE 1B 3B hetero (llama)
    - sc137: DRAKE 1B 3B 8B hetero (llama)
    - sc1137: DRAKE 1B 3B 8B hetero (qwen)
    - sc203: fs-llm hetero (llama)
    - sc262: fed-aya hetero (llama)

- new scenario (hetero client 제대로 설정한 시나리오들)
    - scenario-0~5: DRAKE (sc1 - 3B homo, sc4 - 1B/3B hetero, sc5 - 1B/3B/8B hetero)
    - scenario-7~12: unseen
    - scenario-20~24: HFLB (sc21 - 3B homo, sc23 - 1B/3B hetero)
    - scenario-60~ : fed-aya (sc62)
    - scenario-70~ : fed-llm (sc74)
    - sc103 - DRAKE Qwen hetero


- Large scale NLP 실험:
    - 시나리오: 90 (52 clients), 93 (36 clients)
    - 데이터셋: gdrive files download 1QJK0JtrmrD2AsZvk78WpqT0_TsDHJXVc
        - `dataset` 폴더 안에서 다운 후 `tar -xvf nlp_datasets.tar`
    - 핵심 하이퍼파라미터:
        - lr 1e-4/5e-4 
        - batch size 4
        - cosine scheduler
        - 2 rounds per task
        - 30 rounds (for sft, reduce the iteration for baseline)
        - lora r 16 / lora alpha 32
    - `train_VLM_CL_for_NLP.sh` 참고
    - 테스트셋은 현재 task당 15개씩만 가져오도록 설정되어있음 (eval_VLM_CL.py L655)