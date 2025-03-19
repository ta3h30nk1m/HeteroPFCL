import shutil
import json 
import os
import numpy as np
import random
from collections import defaultdict
import openai
from openai import OpenAI
import time
from multiprocessing import Process, Queue, Manager, Pool, cpu_count
import math


ATTRIBUTE_KEYWORDS = ["color", "size", "shape", "type", "kind", "pattern", "style", "hue", "shade", 
    "tint", "big", "small", "large", "height", "width", "length", "depth", 
    "form", "appearance", "contour", "outline", "texture", "material", "fabric", 
    "surface", "design", "decoration", "print", "motif", "fashion", "aesthetic", 
    "theme", "art style", "brightness", "contrast", "density", "opacity"]

data_type = 'test'
question_file_path = f"MultipleChoice_abstract_v002_{data_type}2015_questions.json"
answer_file_path = f"abstract_v002_{data_type}2015_annotations.json"
data_path = "dataset/Abstract"
max_num = 10000 if data_type == 'train' else 1000


NUM_SECONDS_TO_SLEEP = 0.01
NUM_PROCESSES = 10 #cpu_count()  # Use all available CPU cores

# Set up OpenAI client
client = OpenAI(
    api_key=""
    )

rule = {
    "role": "Assistant",
    "prompt": "Please generate three answer choices for the given question asking about a image referring the given correct answer and the given candidates.\n"
              "The possible answers are the variation of correct and wrong answers. You should use it to infer about the image.\n"
              "You should:\n"
              "1. Select valid choices from the given candidates. If fewer than three valid choices exist, generate additional choices to make a total of three.\n"
              "2. NOT INCLUDE correct answer in the generated choices.\n"
              "3. generate choices relevant to the question, but should not be confusing with the correct answer. Specifically, choices and answers should not be synonyms, nor should they have a relationship of inclusion.\n"
              "You must return the choices by strictly following this format:\"[[Choice A | Choice B | Choice C]]\", for example: \"Choice list: [[red | blue | pink]]\"."
}

def get_eval(content: str, max_tokens: int):
    user_content = [
                        {"type":"text", "text":content['text']},
                    ]
    
    while True:
        try:
            response = client.chat.completions.create(
                messages=[{
                    'role': 'system',
                    'content': 'You are a helpful and precise assistant to generate a choice list.'
                }, {
                    'role': 'user',
                    'content': user_content
                }],
                model="gpt-4o-mini-2024-07-18",
                temperature=0.2,
                max_tokens=max_tokens,
            )
            break
        except openai.APIConnectionError as e:
            print(f"Process {os.getpid()}: The server could not be reached")
            print(e.__cause__)
        except openai.RateLimitError as e:
            print(f"Process {os.getpid()}: A 429 status code was received; backing off")
            time.sleep(5)  # Back off for longer on rate limit errors
        except openai.APIStatusError as e:
            print(f"Process {os.getpid()}: Another non-200-range status code was received")
            print(e.status_code)
            print(e.response)
        time.sleep(NUM_SECONDS_TO_SLEEP)
    return response.choices[0].message.content

def parse_score(review):
    try:
        score = review.split('[[')
        assert len(score) == 2
        score = score[-1].split(']]')[0]
        return score
    except Exception as e:
        print(f"Process {os.getpid()}: Error parsing score: {e}")
        print('error', review)
        return -1

def make_json_files(sampled_data):
    json_data = []
    for idx, data in enumerate(sampled_data):
        new_data = {}
        new_data['id'] = idx
        new_data['image'] = data['image']
        shutil.copy(os.path.join(data_path, f"{data_type}_images", f"abstract_v002_{data_type}2015_{str(data['image_id']).zfill(12)}.png"), os.path.join(data_path, "images", f"abstract_v002_{data_type}2015_{str(data['image_id']).zfill(12)}.png"))

        new_data['conversations'] = [
            {
                "from": "human",
                "value": f"Please respond accurately to the following query. You must choose your answer from the Choice List.<image>Question: {data['question']}\nChoice list:[{data['candidates']}]. Your answer is:"
            },
            {
                "from": "gpt",
                "value": data['answer']
            }
        ]
        json_data.append(new_data)
    return json_data

def remove_multiple_elements(lst, elements_to_remove):
    return [x for x in lst if x not in elements_to_remove]

with open(f"{data_path}/{question_file_path}", "r", encoding="utf-8") as file:
    question_datas = json.load(file)

with open(f"{data_path}/{answer_file_path}", "r", encoding="utf-8") as file:
    answer_datas = json.load(file)

print(len(question_datas['questions']))
print(len(answer_datas['annotations']))

datalists = defaultdict(list)
os.makedirs(os.path.join(data_path, "images"), exist_ok=True)
os.makedirs(os.path.join(data_path, data_type), exist_ok=True)
for question_data, answer_data in zip(question_datas['questions'], answer_datas['annotations']):
    new_data = {}
    if str(question_data['question_id']) != str(answer_data['question_id']):
        raise ValueError("question id different!")
    new_data['question'] = question_data['question']
    new_data['answer'] = answer_data['multiple_choice_answer']
    question_data['multiple_choices'].remove(new_data['answer'])
    new_data['image_id'] = question_data['image_id']
    new_data['image'] = os.path.join(data_path, "images", f"abstract_v002_{data_type}2015_{str(question_data['image_id']).zfill(12)}.png")


    content = {
        "text": (f"[Instruction]\n{rule['prompt']}\n\n"
                f"[Question]\n{question}\n\n"
                f"[Candidates]\n{question_data['multiple_choices']}\n\n"
                f"[Correct Answer]\n{answer}\n")
        "image":
            base64_imgs
        }
    review = get_eval(content, 512)
    sampled_candidates = parse_score(review).split(' | ')
    sampled_candidates += [answer]
    random.shuffle(sampled_candidates)

    if answer_data['answer_type'] == 'number':
        answer_type = 'number'
        # candidates = [x for x in question_data['multiple_choices'] if x.isdigit()]  
    elif answer_data['answer_type'] == 'yes/no':
        # candidates = ['yes'] if new_data['answer']=='no' else ['no']
        answer_type = 'yes/no'
    else:
        # candidates = [x for x in question_data['multiple_choices'] if not x.isdigit()]
        # if 'yes' in candidates:
        #     candidates.remove('yes')
        # if 'no' in candidates:
        #     candidates.remove('no')
        
        answer_type = 'remaining'
        for keyword in ATTRIBUTE_KEYWORDS:
            if keyword in question_data['question']:
                answer_type = 'attribute'

    # sampled_candidates = [new_data['answer']] + random.sample(candidates, min(3, len(candidates)))
    # random.shuffle(sampled_candidates)
    new_data['candidates'] = sampled_candidates
    datalists[answer_type].append(new_data)

'''
os.makedirs(os.path.join(data_path, data_type), exist_ok=True)
for task_num, task_type in enumerate(list(datalists.keys())):
    json_data = []
    sampled_data = np.random.choice(datalists[task_type], size=min(max_num, len(datalists[task_type])), replace=False).tolist()
    for idx, data in enumerate(sampled_data):
        new_data = {}
        new_data['id'] = idx
        new_data['image'] = data['image']
        shutil.copy(os.path.join(data_path, f"{data_type}_images", f"abstract_v002_{data_type}2015_{str(data['image_id']).zfill(12)}.png"), os.path.join(data_path, "images", f"abstract_v002_{data_type}2015_{str(data['image_id']).zfill(12)}.png"))

        new_data['conversations'] = [
            {
                "from": "human",
                "value": f"Please respond accurately to the following query. You must choose your answer from the Choice List.<image>Question: {data['question']}\nChoice list:[{data['candidates']}]. Your answer is:"
            },
            {
                "from": "gpt",
                "value": data['answer']
            }
        ]
        json_data.append(new_data)
    print(f"{task_type} sampled: {len(json_data)}, original: {len(datalists[task_type])}")
    with open(os.path.join(data_path, data_type, f"dataset-{task_num}.json"), 'w') as json_file:
        json.dump(json_data, json_file, indent=4)
'''
task_index = 4
os.makedirs(os.path.join(data_path, data_type), exist_ok=True)
for task_num, task_type in enumerate(list(datalists.keys())):
    if task_type == 'other':
        data_color = [data for data in datalists[task_type] if 'color' in data['question']]
        data_remaining = [data for data in datalists[task_type] if 'color' not in data['question']]
        sampled_data_color = np.random.choice(data_color, size=min(max_num, len(data_color)), replace=False).tolist()
        sampled_data_remaining = np.random.choice(data_remaining, size=min(max_num, len(data_remaining)), replace=False).tolist()
        sampled_json_data_color = make_json_files(sampled_data_color)
        sampled_json_data_remaining = make_json_files(sampled_data_remaining)

        with open(os.path.join(data_path, data_type, f"dataset-{task_index}.json"), 'w') as json_file:
            json.dump(sampled_json_data_color, json_file, indent=4)

        task_index += 1

        with open(os.path.join(data_path, data_type, f"dataset-{task_index}.json"), 'w') as json_file:
            json.dump(sampled_json_data_color, json_file, indent=4)

        task_index += 1

        print(f"{task_type} color sampled: {len(sampled_json_data_color)}, original: {len(data_color)}")
        print(f"{task_type} remaining sampled: {len(sampled_json_data_remaining)}, original: {len(data_remaining)}")
    else:
        json_data = []
        sampled_data = np.random.choice(datalists[task_type], size=min(max_num, len(datalists[task_type])), replace=False).tolist()
        json_data = make_json_files(sampled_data)

        print(f"{task_type} sampled: {len(json_data)}, original: {len(datalists[task_type])}")
        with open(os.path.join(data_path, data_type, f"dataset-{task_index}.json"), 'w') as json_file:
            json.dump(json_data, json_file, indent=4)
        task_index += 1
