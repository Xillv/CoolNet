from hashlib import new
import os
import shutil
import emoji
import re
from transformers import (BertModel, BertTokenizer,
                        RobertaModel,RobertaTokenizer,
                        BertForSequenceClassification,
                        RobertaForSequenceClassification)
import torch
import torch.utils.data as data    
import json



def load_data(dataset, tokenizer, mode='train'):
    assert dataset in ['twitter2015', 'twitter2017']
    data_file = os.path.join('datasets', dataset, mode + '.txt')
    '''
    Former Bridgecorp boss $T$ will be released from jail next month .
    Rod Petricevic
    -1
    280704.jpg
    '''
    with open(data_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        data = []
        for i in range(0, len(lines), 4):
            sentence = lines[i].strip()
            entity = lines[i + 1].strip().replace(' ','')
            
            #sentence = sentence.replace('$T$', entity)
            sentence = emoji.get_emoji_regexp().sub(r'', sentence)
            new_s = ""
            for c in sentence:
                if not c.isprintable():
                    c = ' '
                if c != ' ' or not new_s.endswith(' '):
                    new_s += c
            new_s = new_s.strip()
            
            sentence = new_s.split()
            sentence.append(entity)
            indexed_tokens = tokenizer(new_s, entity)['input_ids']
            mapping = [-1] * len(indexed_tokens)
            sep_cls_id = tokenizer.convert_tokens_to_ids([tokenizer.cls_token, tokenizer.sep_token])
            tmp_id = 0
            if isinstance(tokenizer, BertTokenizer):
                for i, word in enumerate(sentence):
                    while tmp_id < len(indexed_tokens) and indexed_tokens[tmp_id] in sep_cls_id:
                        tmp_id += 1
                    index = tokenizer(word)['input_ids'][1: -1]
                   # print(tokenizer.convert_ids_to_tokens(index))
                    for x in index:
                        assert indexed_tokens[tmp_id] == x, '{}  \n {} \n {}'.format(index, indexed_tokens, word)
                        mapping[tmp_id] = i 
                        tmp_id += 1
                while tmp_id < len(indexed_tokens) and indexed_tokens[tmp_id] in sep_cls_id:
                    tmp_id += 1      
                assert tmp_id == len(indexed_tokens)
            elif isinstance(tokenizer, RobertaTokenizer):
                cnt = -1
                last_is_sep = False
                for i, index in enumerate(indexed_tokens):
                
                    if index in sep_cls_id:
                        last_is_sep = True                        
                        continue
                    if tokenizer.convert_ids_to_tokens([index])[0].startswith('Ġ') or last_is_sep:
                        cnt += 1
                    last_is_sep = False
                    mapping[i] = cnt
                assert cnt + 1 == len(sentence)

            
            data.append((sentence, indexed_tokens, mapping))       
    return data

def load_imgtext(text_file, json_file, tokenizer):
    with open(json_file, 'r') as f1:
        img_to_text = json.load(f1)
    with open(text_file, 'r') as f2:
        lines = f2.readlines()
    dataset = []
    for i in range(0, len(lines), 4):
        imgae_name = lines[i + 3].strip()
        dataset.append(img_to_text[imgae_name])
    res = []
    for sentence in dataset:
        words = sentence.split()
        sentence = ' '.join(words)
       
        sep_cls_id = tokenizer.convert_tokens_to_ids([tokenizer.cls_token, tokenizer.sep_token])
        indexed_tokens = tokenizer(sentence)['input_ids']
        mapping = [-1] * len(indexed_tokens)
        tmp_id = 0
        if isinstance(tokenizer, BertTokenizer):
            for i, word in enumerate(words):
                while tmp_id < len(indexed_tokens) and indexed_tokens[tmp_id] in sep_cls_id:
                    tmp_id += 1
                index = tokenizer(word)['input_ids'][1: -1]
                for x in index:
                    assert indexed_tokens[tmp_id] == x, '{}  \n {} \n {}'.format(index, indexed_tokens, word)
                    mapping[tmp_id] = i 
                    tmp_id += 1
            while tmp_id < len(indexed_tokens) and indexed_tokens[tmp_id] in sep_cls_id:
                tmp_id += 1      
            assert tmp_id == len(indexed_tokens)
        elif isinstance(tokenizer, RobertaTokenizer):
            cnt = -1
            last_is_sep = False
            for i, index in enumerate(indexed_tokens):
                if index in sep_cls_id:
                    last_is_sep = True                        
                    continue
                if tokenizer.convert_ids_to_tokens([index])[0].startswith('Ġ') or last_is_sep:
                    cnt += 1
                last_is_sep = False
                mapping[i] = cnt
            assert cnt + 1 == len(words)
        res.append([words, indexed_tokens, mapping])
    return res
        
    
    

class TwitterDataset(data.Dataset):
    
    def __init__(self, tokenizer, text1, text2=None, labels=None):
        super().__init__()
        if text2:
            self.text = tokenizer(text1, text2,  padding=True)
        else:
            self.text = tokenizer(text1, padding=True)
        self.labels = labels
        
    def __getitem__(self, index):
         item = {key: torch.tensor(val[index]) for key, val in self.text.items()}
         if self.labels:
            item['labels'] = torch.tensor(self.labels[index])
         return item
    
    def __len__(self):
        return len(self.text)

def load_image_dataset(dataset, tokenizer, mode='train'):
    text_file = os.path.join('datasets', dataset, mode + '.txt')
    json_file = os.path.join('datasets', 'captions', '{}_images.json'.format(dataset))
    return load_imgtext(text_file, json_file, tokenizer)



def split_images():
    mode_list = ['train', 'test', 'dev']
    dataset_list = ['twitter2015', 'twitter2017']
    for dataset in dataset_list:
        for mode in mode_list:
            text_file = os.path.join('datasets', dataset, mode + '.txt')
            with open(text_file) as f:
                lines = f.readlines()
            tar_dir = f'datasets/vig_{dataset}/{mode}'
            for label in range(3):
                dir = os.path.join(tar_dir, str(label))
                if not os.path.exists(dir):
                    os.makedirs(dir)        
            for i in range(0, len(lines), 4):
                label = int(lines[i + 2]) + 1
                img_name = lines[i + 3].strip()
                src = os.path.join(f'datasets/{dataset}_images', img_name)
                tar = os.path.join(tar_dir, str(label))
                shutil.copy(src, tar)

def loda_pic_path(dataset, mode='train'):
    text_file = os.path.join('datasets', dataset, mode + '.txt')
    with open(text_file) as f:
        lines = f.readlines()
    pic_path = []
    for i in range(0, len(lines), 4):
        img_name = lines[i + 3].strip()
        src = os.path.join(f'datasets/{dataset}_images', img_name)
        pic_path.append(src)
    return pic_path


def load_labels(dataset, mode='train'):
    text_file = os.path.join('datasets', dataset, mode + '.txt')
    with open(text_file) as f:
        lines = f.readlines()
    labels = []
    for i in range(0, len(lines), 4):
        label = int(lines[i + 2]) + 1
        labels.append(label)
    return labels 
        

if __name__ == '__main__':


    split_images()
