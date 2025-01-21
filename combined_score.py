import jsonlines
import json
import csv
from collections import defaultdict
import os

mode = 'llama3'
Method='llama3_8b_zeroshot'

num_rounds = 20
is_client = True

scenario_num = 0
with open(f"./scenarios/scenario-{scenario_num}.json") as fp:
    scenario = json.load(fp)

scores = {}
client_scores = defaultdict(list)

for client_data in scenario:
    id = client_data['client_id']
    for data in client_data['datasets']:
        data_name = f"{data['dataset']}-{data['subset_id']}"
        
        if is_client:
            with open(f'./eval_results/{mode}/{Method}/client{id}_round{num_rounds}_{data_name}.json', 'r') as fp:
                result = json.load(fp)[-1]
        else:
            with open(f'./eval_results/{mode}/{Method}/server_round{num_rounds}_{data_name}.json', 'r') as fp:
                result = json.load(fp)[-1]
                
        if data['type'] == 'multi-choice':
            score = result['accuracy']
        elif data['type'] == 'open-ended':
            if data['metric'] == 'F1':
                score = 2*(result['precision']*result['recall']) / (result['precision'] + result['recall'])
            elif data['metric'] == 'RougeL':
                score = result['ROUGE_L'][0]
        
        scores[data_name] = score
        client_scores[id].append(score)

avg_score = sum(scores.values()) / len(scores)
client_avg_scores = {id: sum(scores) / len(scores) for id, scores in client_scores.items()}

# Prepare data for CSV
csv_data = [Method, avg_score]
csv_data.extend([client_avg_scores.get(i, '') for i in range(10)])  # Assuming client IDs are 1-10
csv_data.extend(scores.values())

# Prepare header
header = ['method', 'final score']
header.extend([f'client {i}' for i in range(10)])
header.extend(scores.keys())

# Write to CSV
csv_file = 'results.csv'
file_exists = os.path.isfile(csv_file)

with open(csv_file, 'a', newline='') as csvfile:
    writer = csv.writer(csvfile)
    if not file_exists:
        writer.writerow(header)
    writer.writerow(csv_data)

print(f"Method: {Method}")
print(f"Score: {scores}")
print(f"Avg Score: {avg_score}")
print(f"Results have been appended to {csv_file}")