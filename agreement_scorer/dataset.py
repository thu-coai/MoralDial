import torch
from torch.utils.data import Dataset
import pandas as pd
import json
import random


class collate_fn_cls:
    def __init__(self, tokenizer, max_len=128):
        self.tokenizer = tokenizer
        self.max_len = max_len

    def collate_fn(self, batch):
        text1 = [b[0] for b in batch]
        text2 = [b[1] for b in batch]
        labels = [b[2] for b in batch]
        labels = torch.LongTensor(labels)
        inputs = self.tokenizer(text = text1, text_pair = text2, padding='max_length', truncation=True, max_length = self.max_len, return_tensors='pt')
        return inputs, labels


def lower_first(text):
    if text.startswith("I'") or text.startswith('I '):
        return text
    text = text[0].lower() + text[1:]
    return text

class MICDataset(Dataset):
    def __init__(self, path, split, maxsize=-1, remove_Q=False):
        assert(split in ['train','dev','test'])
        nrows = None if maxsize==-1 else maxsize
        df = pd.read_csv(path, nrows=nrows, dtype={'A_agrees':str})
        df = df[df['split']==split]
        self.split = split
        self.text1, self.text2, self.labels = self.process(df, remove_Q=remove_Q)
        assert(len(self.text1) == len(self.text2))
        assert(len(self.text1) == len(self.labels))
        print(self.text1[:10], self.text2[:10], self.labels[:10])
    
    def get_aug(self):
        with open('../data/scorer_data/aug_data.json','r') as f:
            data = json.load(f)
        rot2aug = {}
        for d in data:
            rot2aug[(d['split'], d['original'])] = d['aug']
        return rot2aug

    def process(self, df, remove_Q=False):
        text1s = []
        text2s = []
        labels = []
        error_cnt = 0
        with open('../data/scorer_data/black_rot.txt','r') as f:
            texts = f.read()
        black_rots = eval(texts.replace('\n',''))
        black_rots = set(black_rots)

        rot2aug = self.get_aug()

        random.seed(42)
        cnt = 0
        for _, row in df.iterrows():
            if remove_Q:
                text1 = row['Q']
            else:
                text1 = row['QA']
            text2 = row['rot']
            if text2 in black_rots:
                continue
            label = row['A_agrees']
            if len(label)!=1 or pd.isna(text1) or pd.isna(text2) or pd.isna(label):
                error_cnt+=1
                continue
            
            text1s.append(text1)
            text2s.append(text2)
            labels.append(int(label))
            #labels.append(int(int(label)>0))
        # negative sampling
        shift = 42
        tmp1 = text1s[shift:] + text1s[:shift]
        tmp2 = text2s
        tmp_labels = [1 for _ in range(len(text1s))]
        zipped = [(x,y,z) for x,y,z in zip(tmp1,tmp2,tmp_labels)]
        zipped = random.sample(zipped, int(0.3*len(text1s)))

        for text1, text2, label in zipped:
            text1s.append(text1)
            text2s.append(text2)
            labels.append(label)
        for _, row in df.iterrows():
            text1 = row['QA']
            text2 = row['rot']
            if text2 in black_rots:
                continue
            label = row['A_agrees']
            if len(label)!=1 or pd.isna(text1) or pd.isna(text2) or pd.isna(label):
                continue
            if (self.split, text2) in rot2aug:
                if remove_Q:
                    fake_QA = rot2aug[(self.split,text2)]
                else:
                    fake_QA = 'Q: '+row['Q'].strip() + ' A: '+rot2aug[(self.split,text2)]
                text1s.append(fake_QA)
                text2s.append(text2)
                labels.append(1)
                cnt += 1

            rd = random.random()
            #print(rd)
            if rd<0.02:
                fake_QA = 'Q: '+row['Q'].strip() + ' A: I don\'t know.'
                text1s.append(fake_QA)
                text2s.append(text2)
                labels.append(1)
                cnt += 1
            elif rd<0.03:
                fake_QA = 'Q: '+row['Q'].strip() + ' A:  '
                text1s.append(fake_QA)
                text2s.append(text2)
                labels.append(1)
                cnt += 1
            elif rd<0.04:
                fake_QA = 'Q: '+row['Q'].strip() + ' A: I don\'t know. but ' + lower_first(row['A'])
                text1s.append(fake_QA)
                text2s.append(text2)
                labels.append(int(label))
                cnt += 1
        print('ignore {} samples'.format(error_cnt))
        print('total cnt: {}'.format(len(labels)))
        print('label_distribution: ', labels.count(0), labels.count(1), labels.count(2))
        print('back trans count: {}'.format(cnt))
        return text1s, text2s, labels

    def __getitem__(self, idx):
        return self.text1[idx], self.text2[idx], self.labels[idx]

    def __len__(self):
        return len(self.labels)