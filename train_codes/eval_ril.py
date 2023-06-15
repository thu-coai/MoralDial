import json
from utils import agreement_metric
from transformers import AutoTokenizer, RobertaForSequenceClassification


Bases = ['Based on the rule of thumb, I want to ask another question. ', "Yes, and based on that, here comes another question. ", "I have a similar question for you. ",
"How about this similar question. ", "May you answer the similar question for me? ", "Given what you have learnt, can you answer this question? "]

agreement_tokenizer = AutoTokenizer.from_pretrained("roberta-base")
agreement_model = RobertaForSequenceClassification.from_pretrained('roberta_agreement_model')
device = 'cuda'
agreement_model.to(device)
agreement_model.eval()

def stripstart(ans):
    for base in Bases:
        if len(base)>0 and ans.startswith(base):
            ans = ans[len(base):]
            ans = ans[0].upper() + ans[1:]
            break
    return ans



import numpy as np
dirname = 'out/ril'
import os

logfile = open('logs/ril','w')
filenames = os.listdir(dirname)
# import sys
# logfile = sys.stdout

saves = []
for filename in filenames:
    print(filename)
    path = dirname+'/'+filename
    with open(path,'r') as f:
        data = json.load(f)
    sessions = {'valid':[],'test':[]}
    all_rots = {'valid':[], 'test':[]}
    for key in data:
        for d in data[key]:
            sessions[key] += d['sess']
            all_rots[key] += d['rot']
    for key in sessions:
        sesses = sessions[key]
        rots = all_rots[key]
        Q = [stripstart(sess[-2]) for sess in sesses]
        #Q = [sess[-2] for sess in sesses]
        A = [sess[-1] for sess in sesses]

        ril_scores = agreement_metric(agreement_model, agreement_tokenizer, Q, A, rots, device)

        for sess, ril_score in zip(sesses, ril_scores):
            saves.append({'sess':sess, 'ril_score':ril_score})
        
        print('model name: {}'.format(path), file=logfile)
        print('set: {}'.format(key), file=logfile)
        print('ril_score: {}'.format(np.mean(ril_scores)), file=logfile)
logfile.close()
# with open('logs/ril_sess_log.json','w') as f:
#     json.dump(saves, f, indent=0)