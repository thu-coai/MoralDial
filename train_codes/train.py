from transformers import AutoTokenizer, AdamW, RobertaForSequenceClassification, get_linear_schedule_with_warmup
import torch
from torch.utils.data import DataLoader
import json
from tqdm import tqdm
from datetime import datetime
from dataset import collate_fn_cls
import argparse
from utils import eval_mr, eval_ril, eval_me
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True, type=str, choices=['blenderbot','dialogpt'])
parser.add_argument('--lr', type=float, default=2e-5)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--n_epochs', type=int, default=2)
parser.add_argument('--max_grad_norm', type=float, default=1.0)
parser.add_argument('--log_step', type=int, default=10)
parser.add_argument('--eval_step', type=int, default=300)
parser.add_argument('--save_path', type=str, default='auto')
parser.add_argument('--wo_pt',action='store_true')
parser.add_argument('--gd', action='store_true')
parser.add_argument('--ma', action='store_true')
parser.add_argument('--me', action='store_true')
parser.add_argument('--mr', action='store_true')
parser.add_argument('--ril', action='store_true')
parser.add_argument('--max_size', type=int, default=-1)
args = parser.parse_args()

lr = args.lr
batch_size = args.batch_size
n_epochs = args.n_epochs
max_grad_norm = args.max_grad_norm
log_step = args.log_step
eval_step = args.eval_step


if args.save_path == 'auto':
    save_path = 'models/'+args.model + '_gd'*args.bst+'_ma'*args.ma+'_me'*args.me+'_mr'*args.mr+\
        '_ril'*args.ril+'_wopt'*args.wo_pt+'_'+str(datetime.now().month)+'_'+str(datetime.now().day)
print(save_path)
tmp_name = args.model + '_gd'*args.gd+'_ma'*args.ma+'_me'*args.me+'_mr'*args.mr+\
        '_ril'*args.ril+'_wopt'*args.wo_pt+'_'+str(datetime.now().month)+'_'+str(datetime.now().day)

writer = SummaryWriter('runs/{}'.format(tmp_name))

def train(model, train_dataloader, valid_dataloader, device, optimizer, scheduler):
    step = 1
    best_loss = 1e6
    best_score = 10000
    train_losses = []
    for epoch in range(n_epochs):
        print('epoch {}'.format(epoch + 1))
        now = datetime.now()
        print('time: {}'.format(now))
        for inputs in train_dataloader:
            inputs = {key:val.to(device) for key,val in inputs.items()}
            outputs = model(**inputs)
            loss = outputs.loss
            train_losses.append(loss.item())
            if step%log_step==0:
                print('{} epoch: {}, step:{},  loss: {}'.format(datetime.now(), epoch+1, step, sum(train_losses)/len(train_losses)))
                train_losses = []
            if step%eval_step==0:
                eval_loss = eval(model, valid_dataloader, device)
                model.train()
                if eval_loss<best_loss:
                    best_loss = eval_loss
                    model.save_pretrained(save_path)
                print('{} eval loss: {}, best_loss:{}'.format(datetime.now(), eval_loss, best_loss))
                
                
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            step+=1
    eval_loss = eval(model, valid_dataloader, device)
    if eval_loss<best_loss:
        best_loss = eval_loss
        model.save_pretrained(save_path)
    print('{} eval loss: {}, best_loss:{}'.format(datetime.now(), eval_loss, best_loss))

def eval(model, valid_dataloader, device):
    losses = []
    model.eval()
    with torch.no_grad():
        for inputs in valid_dataloader:
            inputs = {key:val.to(device) for key,val in inputs.items()}
            outputs = model(**inputs)
            loss = outputs.loss
            losses.append(loss.item())
    return sum(losses)/len(losses)


collate_class = collate_fn_cls()


if args.model=='blenderbot':
    from transformers import BlenderbotForConditionalGeneration, BlenderbotTokenizer
    from dataset import Dial_Dataset_Blenderbot as Dial_Dataset
    tokenizer = BlenderbotTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
    if args.wo_pt:
        model = BlenderbotForConditionalGeneration.from_pretrained("facebook/blenderbot-400M-distill")
    else:
        model = BlenderbotForConditionalGeneration.from_pretrained("Blender_pretrain_model")
    collate_fn = collate_class.collate_fn_seq2seq
else:
    from transformers import GPT2LMHeadModel, AutoTokenizer
    from dataset import Dial_Dataset_DialoGPT as Dial_Dataset
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    if args.wo_pt:
        model = GPT2LMHeadModel.from_pretrained("microsoft/DialoGPT-medium")
    else:
        model = GPT2LMHeadModel.from_pretrained("DialoGPT_pretrain_model")
    collate_fn = collate_class.collate_fn

paths = []
if args.gd:
    paths.append('../data/training_data/bst_{}.json')
    paths.append('../data/training_data/dd_{}.json')
if args.ma:
    paths.append('../data/training_data/ma_{}.json')
if args.me:
    paths.append('../data/training_data/me_{}.json')
if args.mr:
    paths.append('../data/training_data/mr_{}.json')
if args.ril:
    paths.append('../data/training_data/ril_{}.json')


train_paths = [path.format('train') for path in paths]
train_dataset = Dial_Dataset(tokenizer=tokenizer, paths=train_paths, max_len = 128, maxsize=args.max_size)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn = collate_fn)

valid_paths = [path.format('valid') for path in paths]
valid_dataset = Dial_Dataset(tokenizer=tokenizer, paths=valid_paths, max_len = 128, maxsize=args.max_size)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn = collate_fn)

test_paths = [path.format('test') for path in paths]
test_dataset = Dial_Dataset(tokenizer=tokenizer, paths=test_paths, max_len = 128, maxsize=args.max_size)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn = collate_fn)

me_paths = ['../data/mic_question_rot/test.json', '../data/mic_question_rot/valid.json']

me_test_data = []
for path in me_paths:
    with open(path, 'r') as f:
        me_test_data += json.load(f)
test_questions = [d['Q'] for d in me_test_data]

import random
random.seed(42)
me_questions = sorted(list(set(test_questions)))
me_questions = random.sample(me_questions, 500)

random.seed(52)
mr_idxs = random.sample(list(range(len(test_questions))), 500)
mr_test_rots = [d['rot'] for d in me_test_data]

mr_test_questions = [test_questions[mr_idx] for mr_idx in mr_idxs]
mr_test_rots = [mr_test_rots[mr_idx] for mr_idx in mr_idxs]

ril_paths = ['../data/mic_question_rot/ril_test.json', '../data/mic_question_rot/ril_valid.json']
ril_test_data= []
for path in ril_paths:
    with open(path, 'r') as f:
        ril_test_data += json.load(f)

ril_rots = [d['rot'] for d in ril_test_data]
ril_sessions = [d['sess'] for d in ril_test_data]

def get_scores(model, tokenizer, agreement_model, agreement_tokenizer):
    model.eval()
    safety_score, me_score = eval_me(model, tokenizer, agreement_model, agreement_tokenizer, match_model, me_questions, device)
    mr_score1, mr_score2 = eval_mr(model, tokenizer, agreement_model, agreement_tokenizer, mr_test_questions, mr_test_rots, device)
    ril_score = eval_ril(model, tokenizer, agreement_model, agreement_tokenizer, ril_sessions, ril_rots, device)
    model.train()
    return safety_score, me_score, mr_score1, mr_score2, ril_score  

agreement_tokenizer = AutoTokenizer.from_pretrained("roberta-base")
agreement_model = RobertaForSequenceClassification.from_pretrained('roberta_agreement_model')
device = 'cuda'
agreement_model.to(device)
agreement_model.eval()
model.to(device)
model.train()


with open('../data/mic_question_rot/safety_rot_stoarge.json','r') as f:
    storage=json.load(f)
from simcse import SimCSE
match_model = SimCSE("princeton-nlp/sup-simcse-roberta-base", device='cuda')
match_model.build_index(storage, use_faiss=True, faiss_fast=False, device='cuda', batch_size=64)

num_parameters = 0
parameters = model.parameters()
for parameter in parameters:
    num_parameters += parameter.numel()
print('number of parameters: {}'.format(num_parameters))

total_steps = len(train_dataloader)*n_epochs
print('total steps: {}'.format(total_steps))
warmup_steps = int(0.05*total_steps)

optimizer = AdamW(model.parameters(), lr=lr, correct_bias=True)
scheduler = get_linear_schedule_with_warmup(optimizer,num_warmup_steps=warmup_steps,
                                                            num_training_steps=total_steps)


loss = eval(model, valid_dataloader, device)
print('Before training, the eval loss is {}'.format(loss))

train(model, train_dataloader, valid_dataloader, device=device, optimizer=optimizer, scheduler=scheduler)
