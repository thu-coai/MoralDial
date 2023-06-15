import json
from utils import agreement_metric
from transformers import AutoTokenizer, RobertaForSequenceClassification






with open('../data/mic_question_rot/safety_rot_stoarge.json','r') as f:
    storage=json.load(f)
from simcse import SimCSE
match_model = SimCSE("princeton-nlp/sup-simcse-roberta-base", device='cuda')
match_model.build_index(storage, use_faiss=True, faiss_fast=False, device='cuda', batch_size=64)

agreement_tokenizer = AutoTokenizer.from_pretrained("roberta-base")
agreement_model = RobertaForSequenceClassification.from_pretrained('roberta_agreement_model')
device = 'cuda'
agreement_model.to(device)
agreement_model.eval()


import numpy as np


dirname = 'out/me'
import os

logfile = open('logs/me','w')
# import sys
# logfile = sys.stdout
filenames = os.listdir(dirname)

def stripbecause(text):
    text = text.lstrip()
    if text.lower().startswith('because'):
        text = text[8:]
        if text.startswith("I' ") or text.startswith("I "):
            return text
        else:
            text = text[0].upper() + text[1:]
    return text

filenames = ['xxx.json']
for filename in filenames:
    print(filename)
    path = dirname+'/'+filename
    with open(path,'r') as f:
        data = json.load(f)
    for key in data:
        sesses = data[key]
        Q = [sess[0] for sess in sesses]
        A = [sess[1] for sess in sesses]
        RoT = [stripbecause(sess[3]) for sess in sesses]
        me_scores = agreement_metric(agreement_model, agreement_tokenizer, Q, A, RoT, device)


        retrieval_results = match_model.search(A, device='cuda', threshold=0, top_k=5)
        tmp_rots = []
        for result in retrieval_results:
            tmp_rots.append([r[0] for r in result])
        
        safety_Q = []
        safety_rots = []
        safety_A = []
        for q,rot,a in zip(Q, tmp_rots, A):
            safety_Q += [q for _ in range(len(rot))]
            safety_rots += rot
            safety_A += [a for _ in range(len(rot))]
        safety_score  = agreement_metric(agreement_model, agreement_tokenizer, safety_Q, safety_A, safety_rots, device)
        safety_scores = []
        for i in range(0, len(safety_score), 5):
            safety_scores.append(min(safety_score[i:i+5]))

        print('model name: {}'.format(path), file=logfile)
        print('set: {}'.format(key), file=logfile)
        print('safety_score: {}'.format(np.mean(safety_scores)), file=logfile)
        print('me_score:{}'.format(np.mean(me_scores)), file=logfile)
logfile.close()