import json
import random
import shutil
random.seed(42)

# client 5 #######################################################################################
original_data = json.load(open('dataset/imagecode/train/dataset-1.json','r'))
sampled_data = random.sample(original_data, 5000)
with open('dataset/imagecode/train/dataset-51.json','w') as fp:
    json.dump(sampled_data, fp, indent=4)
original_data = json.load(open('dataset/imagecode/test/dataset-1.json','r'))
sampled_data = random.sample(original_data, 1000)
with open('dataset/imagecode/test/dataset-51.json','w') as fp:
    json.dump(sampled_data, fp, indent=4)


# client 7 #######################################################################################
shutil.copyfile('dataset/SEED-Bench-2/train/dataset-1.json', 'dataset/SEED-Bench-2/train/dataset-51.json')
shutil.copyfile('dataset/SEED-Bench-2/test/dataset-1.json', 'dataset/SEED-Bench-2/test/dataset-51.json')

original_data = json.load(open('dataset/WCVQA/train/dataset-0.json','r'))
sampled_data = random.sample(original_data, 5000)
with open('dataset/WCVQA/train/dataset-50.json','w') as fp:
    json.dump(sampled_data, fp, indent=4)
shutil.copyfile('dataset/WCVQA/test/dataset-0.json', 'dataset/WCVQA/test/dataset-50.json')

shutil.copyfile('dataset/PathVQA/train/dataset-00.json', 'dataset/PathVQA/train/dataset-50.json')
shutil.copyfile('dataset/PathVQA/test/dataset-00.json', 'dataset/PathVQA/test/dataset-50.json')

# client 8 #######################################################################################
shutil.copyfile('dataset/dvqa/train/dataset-0.json', 'dataset/dvqa/train/dataset-50.json')
original_data = json.load(open('dataset/dvqa/test/dataset-0.json','r'))
sampled_data = random.sample(original_data, 1000)
with open('dataset/dvqa/test/dataset-50.json','w') as fp:
    json.dump(sampled_data, fp, indent=4)

shutil.copyfile('dataset/WCVQA/train/dataset-3.json', 'dataset/WCVQA/train/dataset-53.json')
shutil.copyfile('dataset/WCVQA/test/dataset-3.json', 'dataset/WCVQA/test/dataset-53.json')

shutil.copyfile('dataset/DocVQA/train/dataset-3.json', 'dataset/DocVQA/train/dataset-53.json')
shutil.copyfile('dataset/DocVQA/test/dataset-3.json', 'dataset/DocVQA/test/dataset-53.json')

# client 9 #######################################################################################
shutil.copyfile('dataset/TQA/train/dataset-0.json', 'dataset/TQA/train/dataset-50.json')
original_data = json.load(open('dataset/TQA/test/dataset-0.json','r'))
sampled_data = random.sample(original_data, 1000)
with open('dataset/TQA/test/dataset-50.json','w') as fp:
    json.dump(sampled_data, fp, indent=4)
    
shutil.copyfile('dataset/VSR/train/dataset-0.json', 'dataset/VSR/train/dataset-50.json')
shutil.copyfile('dataset/VSR/test/dataset-0.json', 'dataset/VSR/test/dataset-50.json')

shutil.copyfile('dataset/PathVQA/train/dataset-01.json', 'dataset/PathVQA/train/dataset-51.json')
shutil.copyfile('dataset/PathVQA/test/dataset-01.json', 'dataset/PathVQA/test/dataset-51.json')

shutil.copyfile('dataset/DocVQA/test/dataset-2.json', 'dataset/DocVQA/test/dataset-52.json')
original_data = json.load(open('dataset/DocVQA/train/dataset-2.json','r'))
sampled_data = random.sample(original_data, 4000)
with open('dataset/DocVQA/train/dataset-52.json','w') as fp:
    json.dump(sampled_data, fp, indent=4)

# client 10 #######################################################################################
# <new dataset>

shutil.copyfile('dataset/ChartQA/test/dataset-0.json', 'dataset/ChartQA/test/dataset-50.json')
original_data = json.load(open('dataset/ChartQA/train/dataset-0.json','r'))
sampled_data = random.sample(original_data, 8000)
with open('dataset/ChartQA/train/dataset-50.json','w') as fp:
    json.dump(sampled_data, fp, indent=4)

shutil.copyfile('dataset/DocVQA/test/dataset-0.json', 'dataset/DocVQA/test/dataset-50.json')
original_data = json.load(open('dataset/DocVQA/train/dataset-0.json','r'))
sampled_data = random.sample(original_data, 8000)
with open('dataset/DocVQA/train/dataset-50.json','w') as fp:
    json.dump(sampled_data, fp, indent=4)

shutil.copyfile('dataset/SEED-Bench-2/train/dataset-0.json', 'dataset/SEED-Bench-2/train/dataset-50.json')
original_data = json.load(open('dataset/SEED-Bench-2/test/dataset-0.json','r'))
sampled_data = random.sample(original_data, 1000)
with open('dataset/SEED-Bench-2/test/dataset-50.json','w') as fp:
    json.dump(sampled_data, fp, indent=4)

original_data = json.load(open('dataset/WCVQA/train/dataset-2.json','r'))
sampled_data = random.sample(original_data, 8000)
with open('dataset/WCVQA/train/dataset-42.json','w') as fp:
    json.dump(sampled_data, fp, indent=4)
shutil.copyfile('dataset/WCVQA/test/dataset-2.json', 'dataset/WCVQA/test/dataset-42.json')
