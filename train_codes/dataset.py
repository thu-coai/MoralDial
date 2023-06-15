import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import json 
import os
import random

class collate_fn_cls:
    def __init__(self):
        return

    def collate_fn(self, batch):
        input_ids = [torch.LongTensor(b[0]) for b in batch]
        attention_mask = [torch.LongTensor(b[1]) for b in batch]
        labels = [torch.LongTensor(b[2]) for b in batch]
        input_ids = torch.stack(input_ids, dim=0)
        attention_mask = torch.stack(attention_mask, dim=0)
        labels = torch.stack(labels, dim=0)
        return {'input_ids':input_ids, 'attention_mask':attention_mask, 'labels':labels}
        #return input_ids, attention_mask, labels

    def collate_fn_seq2seq(self, batch):
        input_ids = [torch.LongTensor(b[0]) for b in batch]
        attention_mask = [torch.LongTensor(b[1]) for b in batch]
        decoder_input_ids = [torch.LongTensor(b[2]) for b in batch]
        decoder_attention_mask = [torch.LongTensor(b[3]) for b in batch]
        labels = [torch.LongTensor(b[4]) for b in batch]

        input_ids = torch.stack(input_ids, dim=0)
        attention_mask = torch.stack(attention_mask, dim=0)
        decoder_input_ids = torch.stack(decoder_input_ids, dim=0)
        decoder_attention_mask = torch.stack(decoder_attention_mask, dim=0)
        labels = torch.stack(labels, dim=0)
        return {'input_ids':input_ids, 'attention_mask':attention_mask, 'decoder_input_ids':decoder_input_ids,
                'decoder_attention_mask': decoder_attention_mask, 'labels':labels}
        #return input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, labels

class SC101_RoT_DialoGPT(Dataset):
    def __init__(self, tokenizer, paths, max_len = 128, maxsize=-1):
        data = []
        for path in paths:
            with open(path, 'r') as f:
                data += json.load(f)
        if maxsize!=-1:
            data = random.sample(data, maxsize)
        print(data[0])
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.input_ids, self.attention_masks, self.labels = self.tokenize(data)
        assert(len(self.input_ids)==len(self.attention_masks))
        assert(len(self.input_ids)==len(self.labels))
    
    def truncate(self, sent):
        eos_token = self.tokenizer.eos_token
        response = eos_token + sent + eos_token     
        token_id = self.tokenizer.encode(response, padding=False, truncation=False)
        return token_id
        
        
    def tokenize(self, data):
        input_ids = []
        attention_masks = []
        labels = []
        for sent in tqdm(data):
            token_id = self.truncate(sent)
            if len(token_id)>self.max_len:
                token_id = token_id[:self.max_len]
            input_id = token_id+[self.tokenizer.eos_token_id]*(self.max_len-len(token_id))
            attention_mask = [1]*len(token_id) + [0] * (self.max_len-len(token_id))
            label = token_id + [-100]*(self.max_len-len(token_id))
            input_ids.append(input_id)
            attention_masks.append(attention_mask)
            labels.append(label)
        return input_ids, attention_masks, labels

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attention_masks[idx], self.labels[idx]

    def __len__(self):
        return len(self.input_ids)

class SC101_RoT_Blenderbot(Dataset):
    def __init__(self, tokenizer, paths, max_len = 128, maxsize=-1):
        data = []
        for path in paths:
            with open(path, 'r') as f:
                data += json.load(f)
        if maxsize!=-1:
            data = random.sample(data, maxsize)
        print(data[0])
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.input_ids, self.attention_masks, self.decoder_input_ids, self.decoder_attention_masks, self.labels = self.tokenize(data)
        assert(len(self.input_ids)==len(self.attention_masks))
        assert(len(self.input_ids)==len(self.labels))
    
    def truncate(self, sent):
        bos_token, eos_token = self.tokenizer.bos_token, self.tokenizer.eos_token
        response = bos_token + sent
        token_id = self.tokenizer.encode(response, padding=False, truncation=False)
        return token_id
        
        
    def tokenize(self, data):
        input_ids = []
        attention_masks = []
        decoder_input_ids = []
        decoder_attention_masks = []
        labels = []
        for sent in tqdm(data):
            token_id = self.truncate(sent)
            if len(token_id)>self.max_len:
                token_id = token_id[:self.max_len]
            input_id = [self.tokenizer.eos_token_id] #eos_token is always the end of context
            attention_mask = [1]
            decoder_input_id = token_id+[self.tokenizer.pad_token_id]*(self.max_len-len(token_id))
            decoder_attention_mask = [1]*len(token_id) + [0] * (self.max_len-len(token_id))
            label = token_id[1:] + [-100]*(self.max_len-len(token_id)+1)
            input_ids.append(input_id)
            attention_masks.append(attention_mask)
            decoder_input_ids.append(decoder_input_id)
            decoder_attention_masks.append(decoder_attention_mask)
            labels.append(label)
        return input_ids, attention_masks, decoder_input_ids, decoder_attention_masks, labels

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attention_masks[idx], self.decoder_input_ids[idx], self.decoder_attention_masks[idx], self.labels[idx]

    def __len__(self):
        return len(self.input_ids)


class Dial_Dataset_DialoGPT(Dataset):
    def __init__(self, tokenizer, paths, max_len = 128, maxsize=-1):
        data = []
        for path in paths:
            with open(path, 'r') as f:
                data += json.load(f)
        if maxsize!=-1:
            data = random.sample(data, maxsize)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.input_ids, self.attention_masks, self.labels = self.tokenize(data)
        assert(len(self.input_ids)==len(self.attention_masks))
        assert(len(self.input_ids)==len(self.labels))
    
    def truncate_context(self, sess):
        eos_token = self.tokenizer.eos_token
        response = eos_token + sess[-1] + eos_token     
        response_token_id = self.tokenizer.encode(response, padding=False, truncation=False)
        if len(response_token_id)>self.max_len:
            return None, None
        start = 0
        while start<len(sess):
            context = eos_token.join(sess[start:-1]) 
            context_token_id = self.tokenizer.encode(context, padding=False, truncation=False)
            tmp_len = len(context_token_id)+len(response_token_id)
            if tmp_len<=self.max_len:
                return context_token_id, response_token_id
            start+=1
        
    def tokenize(self, data):
        input_ids = []
        attention_masks = []
        labels = []
        for sess in tqdm(data):
            context_token_id, response_token_id = self.truncate_context(sess)
            if context_token_id is None:
                continue
            tmp_len = len(context_token_id)+len(response_token_id)
            input_id = context_token_id+response_token_id+[self.tokenizer.eos_token_id]*(self.max_len-tmp_len)
            attention_mask = [1]*tmp_len + [0] * (self.max_len-tmp_len)
            label = [-100]*len(context_token_id) + response_token_id + [-100]*(self.max_len-tmp_len)
            input_ids.append(input_id)
            attention_masks.append(attention_mask)
            labels.append(label)
        return input_ids, attention_masks, labels

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attention_masks[idx], self.labels[idx]

    def __len__(self):
        return len(self.input_ids)

class Dial_Dataset_Blenderbot(Dataset):
    def __init__(self, tokenizer, paths, max_len = 128, maxsize=-1):
        data = []
        for path in paths:
            with open(path, 'r') as f:
                data += json.load(f)
        if maxsize!=-1:
            data = random.sample(data, maxsize)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.input_ids, self.attention_masks, self.decoder_input_ids, self.decoder_attention_masks, self.labels = self.tokenize(data)
        assert(len(self.input_ids)==len(self.attention_masks))
        assert(len(self.decoder_input_ids)==len(self.labels))
        assert(len(self.decoder_input_ids)==len(self.decoder_attention_masks))
    

    def truncate_context(self, sess):
        bos_token, eos_token = self.tokenizer.bos_token, self.tokenizer.eos_token
        response = bos_token + sess[-1]
        response_token_id = self.tokenizer.encode(response, padding=False, truncation=False)
        if len(response_token_id)>self.max_len:
            return None, None
        start = 0
        while start<len(sess):
            context = (eos_token+bos_token).join(sess[start:-1]) # automatic add eos_token by following tokenier
            context_token_id = self.tokenizer.encode(context, padding=False, truncation=False)
            tmp_len = len(context_token_id)+len(response_token_id)
            if tmp_len<=self.max_len:
                return context_token_id, response_token_id
            start+=1
        return None, None

    def tokenize(self, data):
        input_ids = []
        attention_masks = []
        decoder_input_ids = []
        decoder_attention_masks = []
        labels = []
        ignored_sample_cnt = 0
        for sess in tqdm(data):
            context_token_id, response_token_id = self.truncate_context(sess)
            if context_token_id is None:
                ignored_sample_cnt += 1
                continue

            input_id = context_token_id+[self.tokenizer.pad_token_id]*(self.max_len-len(context_token_id))
            attention_mask = [1]*len(context_token_id) + [0] * (self.max_len-len(context_token_id))
            
            decoder_input_id = response_token_id + [self.tokenizer.pad_token_id]*(self.max_len-len(response_token_id))
            decoder_attention_mask = [1]*len(response_token_id) + [0]*(self.max_len-len(response_token_id))
            
            label = response_token_id + [-100]*(self.max_len-len(response_token_id))
            # shift label to left
            label = label[1:] + [-100]

            input_ids.append(input_id)
            attention_masks.append(attention_mask)
            decoder_input_ids.append(decoder_input_id)
            decoder_attention_masks.append(decoder_attention_mask)
            labels.append(label)
        print('ignore {} samples'.format(ignored_sample_cnt))
        return input_ids, attention_masks, decoder_input_ids, decoder_attention_masks, labels

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attention_masks[idx], self.decoder_input_ids[idx], self.decoder_attention_masks[idx], self.labels[idx]

    def __len__(self):
        return len(self.input_ids)