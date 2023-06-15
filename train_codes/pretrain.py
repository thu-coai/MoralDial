from transformers import AdamW, Adafactor, get_linear_schedule_with_warmup
import torch
from torch.utils.data import DataLoader
import json
from tqdm import tqdm
from datetime import datetime
from dataset import collate_fn_cls

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True, type=str, choices=['blenderbot','dialogpt','blenderbot-3b','dialogpt-large'])
parser.add_argument('--lr', type=float, default=2e-5)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--n_epochs', type=int, default=5)
parser.add_argument('--max_grad_norm', type=float, default=1.0)
parser.add_argument('--log_step', type=int, default=50)
parser.add_argument('--eval_step', type=int, default=2000)
parser.add_argument('--save_path', type=str, required=True)
parser.add_argument('--fp16', action='store_true', help='mixed precision')
parser.add_argument('--accelerate', action='store_true', help='accelerate')
parser.add_argument('--fp16_opt_level', default='O1', type=str, required=False)

args = parser.parse_args()

lr = args.lr
batch_size = args.batch_size
n_epochs = args.n_epochs
max_grad_norm = args.max_grad_norm
log_step = args.log_step
eval_step = args.eval_step


def train(model, train_dataloader, valid_dataloader, device, optimizer, scheduler):
    step = 1
    best_loss = 1e6
    train_losses = []
    for epoch in range(n_epochs):
        print('epoch {}'.format(epoch + 1))
        now = datetime.now()
        print('time: {}'.format(now))
        for inputs in train_dataloader:
            inputs = {key:val.to(device) for key,val in inputs.items()}
            outputs = model(**inputs)
            loss = outputs.loss


            #logits = outputs.logits
            train_losses.append(loss.item())
            if step%log_step==0:
                print('loss: {}, step:{}'.format(sum(train_losses)/len(train_losses), step))
                train_losses = []
            if step%eval_step==0:
                eval_loss = eval(model, valid_dataloader, device)
                model.train()
                if eval_loss<best_loss:
                    best_loss = eval_loss
                    model.save_pretrained(args.save_path)
                print('eval loss: {}, best_loss:{}'.format(eval_loss, best_loss))
                
            optimizer.zero_grad()
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            step+=1

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


tokenizer, model = None, None

collate_class = collate_fn_cls()
if args.model=='blenderbot':
    from transformers import BlenderbotForConditionalGeneration, BlenderbotTokenizer
    from dataset import SC101_RoT_Blenderbot as SC101_RoT
    tokenizer = BlenderbotTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
    model = BlenderbotForConditionalGeneration.from_pretrained("facebook/blenderbot-400M-distill")
    collate_fn = collate_class.collate_fn_seq2seq
elif args.model=='blenderbot-3b':
    from transformers import BlenderbotForConditionalGeneration, BlenderbotTokenizer
    from dataset import SC101_RoT_Blenderbot as SC101_RoT
    tokenizer = BlenderbotTokenizer.from_pretrained("facebook/blenderbot-3b")
    model = BlenderbotForConditionalGeneration.from_pretrained("facebook/blenderbot-3b")
    collate_fn = collate_class.collate_fn_seq2seq
elif args.model=='dialogpt':
    from transformers import GPT2LMHeadModel, AutoTokenizer
    from dataset import SC101_RoT_DialoGPT as SC101_RoT
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    model = GPT2LMHeadModel.from_pretrained("microsoft/DialoGPT-medium")
    collate_fn = collate_class.collate_fn
else:
    from transformers import GPT2LMHeadModel, AutoTokenizer
    from dataset import SC101_RoT_DialoGPT as SC101_RoT
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large")
    model = GPT2LMHeadModel.from_pretrained("microsoft/DialoGPT-large")
    collate_fn = collate_class.collate_fn


paths = ['../data/training_data/sc101_train.json']
train_dataset = SC101_RoT(tokenizer=tokenizer, paths=paths, max_len = 128, maxsize=100)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn = collate_fn)

paths = ['../data/training_data/sc101_valid.json']
valid_dataset = SC101_RoT(tokenizer=tokenizer, paths=paths, max_len = 128, maxsize=100)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn = collate_fn)

paths = ['../data/training_data/sc101_test.json']
test_dataset = SC101_RoT(tokenizer=tokenizer, paths=paths, max_len = 128, maxsize=-1)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn = collate_fn)


device = 'cuda'

model.to(device)
model.train()

num_parameters = 0
parameters = model.parameters()
for parameter in parameters:
    num_parameters += parameter.numel()
print('number of parameters: {}'.format(num_parameters))

total_steps = len(train_dataloader)*n_epochs//batch_size
warmup_steps = int(0.1*total_steps)

if args.model=='blenderbot-3b':
    optimizer = Adafactor(model.parameters(), lr = lr, relative_step=False)
else:
    optimizer = AdamW(model.parameters(), lr=lr, correct_bias=True)
scheduler = get_linear_schedule_with_warmup(optimizer,num_warmup_steps=warmup_steps,
                                                            num_training_steps=total_steps)

if args.fp16:
    try:
        from apex import amp
    except ImportError:
        raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
    model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

if args.accelerate:
    from accelerate import Accelerator
    accelerator = Accelerator()
    train_dataloader, valid_dataloader, model, optimizer = accelerator.prepare(
        train_dataloader, valid_dataloader, model, optimizer
    )

loss = eval(model, valid_dataloader, device)
print(loss)

train(model, train_dataloader, valid_dataloader, device=device, optimizer=optimizer, scheduler=scheduler)

loss = eval(model, test_dataloader, device)
print(loss)