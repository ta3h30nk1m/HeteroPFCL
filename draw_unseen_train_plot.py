import jsonlines
import json
import csv
from collections import defaultdict
import os
import matplotlib.pyplot as plt

mode_method_dict = {
    # 'Frozen weight (1B)':'sft_bs4_svaeoptim_lr2e-5_sc7_1tasks_1rounds_fixitr1000_memonly',
    # 'Frozen weight (1B, rank32)':'sft_bs4_svaeoptim_lr2e-5_sc7_1tasks_1rounds_fixitr1000_memonly_rank32',
    # 'Frozen weight (3B)':'sft_bs4_svaeoptim_lr2e-5_sc8_1tasks_1rounds_fixitr1000_memonly',
    # 'Frozen weight (3B, rank32)':'sft_bs4_svaeoptim_lr2e-5_sc8_1tasks_1rounds_fixitr1000_memonly_rank32',
    # 'FedAvg(1B)':'sft_bs4_svaeoptim_lr2e-5_sc7_1tasks_1rounds_fixitr1000_fedavg_r15_memonly',
    # 'FedAvg(3B)':'sft_bs4_svaeoptim_lr2e-5_sc8_1tasks_1rounds_fixitr1000_fedavg_r15_memonly',
    # 'FedAvg(3B, rank32)':'sft_bs4_saveoptim_lr2e-5_sc8_4tasks_1rounds_fixitr1000_fedavg_r15_memonly_rank32',
    # 'FedOurs(1B)':'sft_bs4_svaeoptim_lr2e-5_sc7_1tasks_1rounds_fixitr1000_fedours_t0.2_r15_memonly',
    # 'FedOurs(3B)':'sft_bs4_svaeoptim_lr2e-5_sc8_1tasks_1rounds_fixitr1000_fedours_t0.2_r15_memonly',
    # 'FedOurs(3B, rank32)':'sft_bs4_svaeoptim_lr2e-5_sc8_1tasks_1rounds_fixitr1000_fedours_t0.2_rank32_r15_memonly',
    
    # 'Frozen weight (1B)':'sft_bs4_svaeoptim_lr2e-5_sc9_1tasks_1rounds_fixitr1000_memonly',
    # 'Frozen weight (1B, rank32)':'sft_bs4_svaeoptim_lr2e-5_sc9_1tasks_1rounds_fixitr1000_memonly_rank32',
    # 'Frozen weight (3B)':'sft_bs4_svaeoptim_lr2e-5_sc10_1tasks_1rounds_fixitr1000_memonly',
    # 'Frozen weight (3B, rank32)':'sft_bs4_svaeoptim_lr2e-5_sc10_1tasks_1rounds_fixitr1000_memonly_rank32',
    # 'FedAvg(1B)':'sft_bs4_svaeoptim_lr2e-5_sc9_1tasks_1rounds_fixitr1000_fedavg_r15_memonly',
    # 'FedAvg(3B)':'sft_bs4_svaeoptim_lr2e-5_sc10_1tasks_1rounds_fixitr1000_fedavg_r15_memonly',
    # 'FedAvg(3B, rank32)':'sft_bs4_saveoptim_lr2e-5_sc10_4tasks_1rounds_fixitr1000_fedavg_r15_memonly_rank32',
    # 'FedOurs(1B)':'sft_bs4_svaeoptim_lr2e-5_sc9_1tasks_1rounds_fixitr1000_fedours_t0.2_r15_memonly',
    # 'FedOurs(3B)':'sft_bs4_svaeoptim_lr2e-5_sc10_1tasks_1rounds_fixitr1000_fedours_t0.2_r15_memonly',
    # 'FedOurs(3B, rank32)':'sft_bs4_svaeoptim_lr2e-5_sc10_1tasks_1rounds_fixitr1000_fedours_t0.2_rank32_r15_memonly',
}

# colors = ['#3A1730', '#C18A3D', '#588157', '#E63946', '#BCBD22', '#17BECF', '#457B9D']
colors = ['#457B9D', '#314832', '#D8CFC0', '#E63946', '#3A1730', '#C18A3D', '#588157', '#38322C', '#BCBD22', '#17BECF']

scenario_num = 8
with open(f"./scenarios/scenario-{scenario_num}.json") as fp:
    scenario = json.load(fp)

# iters = [0, 25, 50, 75, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800]
iters = [0, 25, 50, 75, 100, 150, 200, 250, 300, 350, 400]
# iters = [0, 5, 10, 20, 30, 40, 50, 100, 150, 200, 250, 300, 350]
# iters = [0, ,5, 10, 20, 30, 40, 50, 100]
num_rounds = 1

# plot_mode = 'seen_task'
data_indices = [
    [0],[0],[0],[0],[0],
    [0],[0],[0],[0],[0],
]

for client_data in scenario:
    id = client_data['client_id']
    mode_scores = {}
    for mode in mode_method_dict.keys():
        Method = mode_method_dict[mode]
        client_scores = []
        for num_round in range(num_rounds):
            data_index = data_indices[num_round]
            done = False
            for iter in iters:
                summed_score = 0
                for d_idx in data_index:
                    data = client_data['datasets'][d_idx]
                    data_name = f"{data['dataset']}-{data['subset_id']}"
                    
                    try:
                        filename = f'./eval_results/{Method.split("_")[0]}/{Method}/client{id}_round{num_round+1}_iter{iter}_{data_name}.json'
                        with open(filename, 'r') as fp:
                            result = json.load(fp)[-1]
                    except Exception as e:
                        print(e)
                        done = True
                        break
                    
                    if data['type'] == 'multi-choice':
                        score = result['accuracy']
                    elif data['type'] == 'open-ended':
                        if data['metric'] == 'F1':
                            score = 2*(result['precision']*result['recall']) / (result['precision'] + result['recall'])
                        elif data['metric'] == 'RougeL':
                            score = result['ROUGE_L'][0]
                        elif data['metric'] == 'cider':
                            score = result['CIDEr'][0]
                    summed_score += score
                if done:
                    break
                client_scores.append(summed_score / len(data_index))                    
        mode_scores[mode] = client_scores
        
    # Plotting the scores
    plt.figure(figsize=(8, 4.8))
    plt.axes().set_facecolor("#F5F5F5")
    plt.rc('axes', labelsize=16)
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)
    
    y = iters#[0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]

    for mode, scores in mode_scores.items():
        print(f'{data_name} | {mode} | AUC: {sum(scores)/len(scores)} | Final Acc: {scores[-1]}')
        plt.plot(y[:len(scores)], scores, label=f'{mode}', linewidth=2.0)#, color=mode_color_dict[mode])#, marker='o')
    
    plt.title(f'{data_name} Scores', fontsize=20)
    plt.xlabel('Iteration', fontsize=18)
    plt.ylabel('Score', fontsize=18)
    plt.legend(fontsize=14)
    # plt.grid(axis='y')
    plt.grid(True)
    

    # Save the plot
    plt.savefig(f'plot_unseen_train_client_{id}_sc{scenario_num}.png')