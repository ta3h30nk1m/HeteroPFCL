import json
import os
import shutil
data_dir = 'dataset/COCOQA'
type_name = 'train'
os.makedirs(os.path.join(data_dir, type_name), exist_ok=True)
os.makedirs(os.path.join(data_dir, 'images'), exist_ok=True)

for task_num in range(4):
    txt_questions = f"{data_dir}/{type_name}_annotations/type_{task_num}/questions.txt"
    txt_img_ids = f"{data_dir}/{type_name}_annotations/type_{task_num}/img_ids.txt"
    txt_answers = f"{data_dir}/{type_name}_annotations/type_{task_num}/answers.txt"
    out_json = f"{data_dir}/{type_name}/dataset-{task_num}.json"

    # 파일 읽기
    with open(txt_questions, "r", encoding="utf-8") as f:
        questions = [line.strip() for line in f.readlines()]

    with open(txt_img_ids, "r", encoding="utf-8") as f:
        img_ids = [line.strip() for line in f.readlines()]

    with open(txt_answers, "r", encoding="utf-8") as f:
        answers = [line.strip() for line in f.readlines()]

    dataset = []
    for idx, (img_id, question, answer) in enumerate(zip(img_ids, questions, answers)):
        entry = {
            "id": idx,
            "image": f"{data_dir}/images/{img_id.zfill(12)}.jpg",
            "conversations": [
                {
                    "from": "human",
                    "value": f"Please respond accurately to the following query. <image>Question: {question} Your answer is:"
                },
                {
                    "from": "gpt",
                    "value": answer
                }
            ]
        }
        dataset.append(entry)
        shutil.copy(f"{data_dir}/{type_name}_images/type_{task_num}/{img_id.zfill(12)}.jpg", f"{data_dir}/images/{img_id.zfill(12)}.jpg")

    # JSON 파일 저장
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False)

