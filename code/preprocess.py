# allenai/tulu-v2-sft-mixture
import json
from datasets import load_dataset
import os
import random
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='allenai/tulu-v2-sft-mixture')
parser.add_argument('--n_cluster', type=int, default=4)
parser.add_argument('--output_file', type=str)
parser.add_argument('--output_cluster', type=str)
args = parser.parse_args()



raw_dataset = load_dataset(args.dataset,split='train')
f = open(args.output_file,'w')
for i in tqdm(range(len(raw_dataset))):
    data = raw_dataset[i]
    newdata = dict()
    if data['messages'][0]['role'] == 'user' and data['messages'][1]['role'] == 'assistant':
        newdata['instruction'] = data['messages'][0]['content']
        newdata['input'] = ''
        newdata['output'] = data['messages'][1]['content']
    elif data['messages'][0]['role'] == 'system' and data['messages'][1]['role'] == 'user' and data['messages'][2]['role'] == 'assistant':
        newdata['instruction'] = data['messages'][1]['content']
        newdata['input'] = ''
        newdata['output'] = data['messages'][2]['content']
    f.write(json.dumps(newdata,ensure_ascii=False)+'\n')
f.close()




assignment =[0]* len(raw_dataset)
for i in range(len(data)):
    assignment[i] = random.choice(list(range(args.n_cluster)))
json.dump(data,open(args.output_cluster,'w'))

