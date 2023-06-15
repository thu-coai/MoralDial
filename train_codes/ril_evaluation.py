from transformers import AutoTokenizer, RobertaForSequenceClassification
import torch
import json
from tqdm import trange
import argparse
from utils import generate_samples_blender, generate_samples_dialogpt, agreement_metric, lowfirst


parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True, type=str, choices=['blenderbot','dialogpt'])
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--model_path',required=True, type=str)

args = parser.parse_args()

batch_size = args.batch_size


if args.model=='blenderbot':
    from transformers import BlenderbotForConditionalGeneration, BlenderbotTokenizer
    from utils import generate_samples_blender as generate_func
    tokenizer = BlenderbotTokenizer.from_pretrained("facebook/blenderbot-400M-distill") 
    model = BlenderbotForConditionalGeneration.from_pretrained(args.model_path)
    #model = BlenderbotForConditionalGeneration.from_pretrained("blenderbot")
else:
    from transformers import GPT2LMHeadModel, AutoTokenizer
    from utils import generate_samples_dialogpt as generate_func
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    #model = GPT2LMHeadModel.from_pretrained("DialoGPT/medium")
    model = GPT2LMHeadModel.from_pretrained(args.model_path)



device = 'cuda'
model.to(device)
model.eval()

import random
import numpy as np


paths = ['../data/mic_question_rot/ril_test.json', '../data/mic_question_rot/ril_valid.json']
data = []
names = ['test','valid']
agreement_scores = []

pos = args.model_path.rfind('/')
save_name = 'out/ril/'+args.model_path[pos+1:]+'.json'
print(save_name)

outputs = {'valid':[], 'test':[]}
for name, path in zip(names, paths):
    print('set: {}'.format(name))
    with open(path, 'r') as f:
        data = json.load(f)
 
    print('There are {} questions'.format(len(data)))

    random.seed(42)

    for i in trange(0, len(data), batch_size):
        batch_data = data[i: min(i+batch_size, len(data))]

        tmp_Q = [d['Q'] for d in batch_data]

        contexts = [d['sess'] for d in batch_data]
        generates = generate_func(model, tokenizer, contexts, device, remove_last=True)

        tmp_rot = [d['rot'] for d in batch_data]

        outputs[name].append({'rot':tmp_rot,'sess':generates})

with open(save_name, 'w') as f:
    json.dump(outputs, f, indent=0)

        