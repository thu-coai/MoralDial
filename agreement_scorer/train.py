from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, get_linear_schedule_with_warmup
import torch
from torch.utils.data import DataLoader
import json
from tqdm import tqdm
from datetime import datetime
from dataset import collate_fn_cls, MICDataset
import argparse
from sklearn.metrics import classification_report, accuracy_score, f1_score

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True, type=str, choices=['albert-base-v2', 'roberta-base', 'bert-base-uncased'])
parser.add_argument('--lr', type=float, default=2e-5)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--n_epochs', type=int, default=5)
parser.add_argument('--max_grad_norm', type=float, default=1.0)
parser.add_argument('--log_step', type=int, default=10)
parser.add_argument('--eval_step', type=int, default=500)
parser.add_argument('--num_labels', type=int, default=3)
parser.add_argument('--save_path', type=str, required=True)
parser.add_argument('--remove_Q',  action='store_true')
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
    best_f1 = 0
    train_losses = []
    preds = []
    all_labels = []
    for epoch in range(n_epochs):
        print('epoch {}'.format(epoch + 1))
        now = datetime.now()
        print('time: {}'.format(now))
        for inputs, labels in train_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(**inputs, labels=labels)
            logits = outputs.logits
            pred = torch.argmax(logits, dim=1)
            preds += pred.tolist()
            all_labels += labels.tolist()

            loss = outputs.loss
            train_losses.append(loss.item())
            if step%log_step==0:
                f1 = f1_score(all_labels, preds, average='macro')
                acc = accuracy_score(all_labels, preds)
                print('epoch:{} step:{} loss:{} acc:{} f1:{}'.format(epoch+1, step, sum(train_losses)/len(train_losses), acc, f1))
                preds = []
                all_labels = []
                train_losses = []
            if step%eval_step==0:
                eval_loss, eval_acc, eval_f1, eval_report = eval(model, valid_dataloader, device)
                model.train()
                if best_f1<eval_f1:
                    best_f1 = eval_f1
                    model.save_pretrained(args.save_path)
                print('eval loss: {}, eval f1: {} eval acc: {}'.format(eval_loss, eval_f1, eval_acc))
                print('best f1: {}'.format(best_f1))
                print(eval_report)
                
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            step+=1

def eval(model, valid_dataloader, device):
    losses = []
    model.eval()
    preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in valid_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            logits = outputs.logits
            pred = torch.argmax(logits, dim=1)

            preds += pred.tolist()
            all_labels += labels.tolist()
            losses.append(loss.item())
    f1 = f1_score(all_labels, preds, average='macro')
    acc = accuracy_score(all_labels, preds)
    report = classification_report(all_labels, preds)
    return sum(losses)/len(losses), acc, f1, report


tokenizer = AutoTokenizer.from_pretrained(args.model) 
model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=args.num_labels)

collate_class = collate_fn_cls(tokenizer=tokenizer, max_len=128)
collate_fn = collate_class.collate_fn


path = '../data/scorer_data/MIC.csv'

train_dataset = MICDataset(path=path, split='train', maxsize=-1, remove_Q=args.remove_Q)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn = collate_fn)

valid_dataset = MICDataset(path=path, split='dev', maxsize=-1,  remove_Q=args.remove_Q)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn = collate_fn)

test_dataset = MICDataset(path=path, split='test', maxsize=-1,  remove_Q=args.remove_Q)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn = collate_fn)

device = 'cuda'

model.to(device)
model.train()

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


loss, acc, f1, report = eval(model, test_dataloader, device)
print(acc)
print(f1)
print(report)