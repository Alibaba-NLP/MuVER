#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   data_loader.py
@Time    :   2021/04/07 16:13:52
@Author  :   Xinyin Ma
@Version :   0.1
@Contact :   maxinyin@zju.edu.cn
'''

import os
import json
import random
from tqdm import tqdm
import numpy as np 

import nltk
import torch
from torch.utils.data import Dataset, Sampler, DataLoader
from torch.utils.data.distributed import DistributedSampler

from muver.utils.params import WORLDS

CLS_TAG = "[CLS]"
SEP_TAG = "[SEP]"
MENTION_START_TAG = "[unused0]"
MENTION_END_TAG = "[unused1]"
ENTITY_TAG = "[unused2]"

class SubworldBatchSampler(Sampler):
    def __init__(self, batch_size, subworld_idx):
        self.batch_size = batch_size
        self.subworld_idx = subworld_idx

    def __iter__(self):
        for world_name, world_value in self.subworld_idx.items():
            world_value['perm_idx'] = torch.randperm(len(world_value['idx']))
            world_value['pointer'] = 0
        world_names = list(self.subworld_idx.keys())

        while len(world_names) > 0:
            world_name = np.random.choice(world_names)
            world_value = self.subworld_idx[world_name]
            start_pointer = world_value['pointer']
            sample_perm_idx = world_value['perm_idx'][start_pointer:start_pointer + self.batch_size]
            sample_idx = [world_value['idx'][idx] for idx in sample_perm_idx]

            if len(sample_idx) > 0:
                yield sample_idx
            
            if len(sample_idx) < self.batch_size:
                world_names.remove(world_name)
            world_value['pointer'] += self.batch_size
    
    def __len__(self):
        return sum([len(value) // self.batch_size + 1 for _, value in self.subworld_idx.items()])
    
class SubWorldDistributedSampler(DistributedSampler):
    def __init__(self, batch_size, subworld_idx, num_replicas, rank):
        self.batch_size = batch_size
        self.subworld_idx = subworld_idx
        self.num_replicas = num_replicas
        self.rank = rank

        self.epoch = 0

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch)

        for world_name, world_value in self.subworld_idx.items():
            world_value['perm_idx'] = torch.randperm(len(world_value['idx']), generator=g).tolist()
            world_value['pointer'] = 0
        world_names = list(self.subworld_idx.keys())

        while len(world_names) > 0:
            world_idx = torch.randint(len(world_names), size=(1, ), generator=g).tolist()[0]
            world_name = world_names[world_idx]
            
            world_value = self.subworld_idx[world_name]
            start_pointer = world_value['pointer']
            sample_perm_idx = world_value['perm_idx'][start_pointer : start_pointer + self.batch_size]

            if len(sample_perm_idx) == 0:
                world_names.remove(world_name)
                continue
            
            if len(sample_perm_idx) < self.batch_size :
                world_names.remove(world_name)
                sample_perm_idx = sample_perm_idx + world_value['perm_idx'][:self.batch_size - len(sample_perm_idx)]
            #print(self.rank, sample_perm_idx)
            sample_perm_idx = sample_perm_idx[self.rank::self.num_replicas]
            
            try:
                sample_idx = [world_value['idx'][idx] for idx in sample_perm_idx]
                assert len(sample_idx) == self.batch_size // self.num_replicas
            except:
                print(world_name, sample_perm_idx, sample_idx, len(world_value['idx']))
            yield sample_idx
            world_value['pointer'] += self.batch_size
        
        self.epoch += 1

    #def __len__(self):
    #    return sum([len(value) // self.batch_size + 1 for _, value in self.subworld_idx.items()])

class EncodeDataset(Dataset):
    def __init__(self, document_path, world, tokenizer, max_seq_len, max_sentence_num, all_sentences = False):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.max_sentence_num = max_sentence_num
        self.all_sentences = all_sentences
        self.world = world
        
        self.seq_lens = {}

        preprocess_path = os.path.join(document_path, 'preprocess_multiview')

        if os.path.exists(preprocess_path) and os.path.exists(os.path.join(preprocess_path, world + '.pt')):
            self.samples, self.entity_title_to_id = torch.load(os.path.join(preprocess_path, world + '.pt'))
            print("World/{}: Load {} samples".format(world, len(self.samples)))
        else:
            if not os.path.exists(preprocess_path):
                os.mkdir(preprocess_path)
            
            document_path = os.path.join(document_path, world + '.json')
            self.samples, self.entity_title_to_id = self.load_entity_description(document_path, tokenizer, world)
            torch.save([self.samples, self.entity_title_to_id], os.path.join(preprocess_path, world + '.pt'))
        
    def __len__(self):
        return len(self.samples)
    
    def get_nth_title(self, idx):
        return self.samples[idx]['title']
    
    def load_entity_description(self, document_path, tokenizer, world):
        entity_desc = []
        entity_title_to_id = {}

        num_lines = sum(1 for line in open(document_path, 'r'))
        print("World/{}: preprocessing {} samples".format(world, num_lines))

        sentence_nums = {}
        with open(document_path, 'r') as f:
            for idx, line in enumerate(tqdm(f, total=num_lines)):
                info = json.loads(line)
                token_ids = self.tokenize_split_description(info['title'], info['text'], tokenizer)
                entity_desc.append({
                    "token_ids": token_ids,
                    "title": info['title']
                })
                num = sentence_nums.get(len(token_ids), 0)
                sentence_nums[len(token_ids)] = num + 1
                entity_title_to_id[info['title']] = idx

        #print(sorted(sentence_nums.items(), key = lambda x: x[0]))
        return entity_desc, entity_title_to_id
    
    def tokenize_description(self, title, desc, tokenizer):
        encode_text = [CLS_TAG] + tokenizer.tokenize(title) + [ENTITY_TAG] + tokenizer.tokenize(desc)
        encode_text = encode_text[:self.max_cand_len - 1] + [SEP_TAG]

        tokens = tokenizer.convert_tokens_to_ids(encode_text)
        if len(tokens) < self.max_cand_len:
            tokens += [0] * (self.max_cand_len - len(tokens))
        
        assert(len(tokens) == self.max_cand_len)
        return tokens 

    def tokenize_split_description(self, title, desc, tokenizer):
        #if not is_split_by_sentence:
        #    encode_text = [CLS_TAG] + tokenizer.tokenize(title) + [ENTITY_TAG] + tokenizer.tokenize(desc)
        #    encode_text = encode_text[:self.max_cand_len - 1] + [SEP_TAG]
        #else:
        title_text = tokenizer.tokenize(title) + [ENTITY_TAG]

        multi_sent = []
        pre_text = []
        for sent in nltk.sent_tokenize(desc.replace(' .', '.')):
            text = tokenizer.tokenize(sent)
            pre_text += text
            if len(pre_text) <= 5:
                continue
            whole_text = title_text + pre_text
            whole_text = [CLS_TAG] + whole_text[:self.max_seq_len - 2] + [SEP_TAG]
            tokens = tokenizer.convert_tokens_to_ids(whole_text)
            pre_text = []

            if len(tokens) < self.max_seq_len:
                tokens += [0] * (self.max_seq_len - len(tokens))
            assert len(tokens) == self.max_seq_len
            multi_sent.append(tokens)

        return multi_sent
    
    def __getitem__(self, idx):
        if self.all_sentences:
            entity_ids = self.samples[idx]['token_ids']
        else:
            entity_ids = self.samples[idx]['token_ids'][:self.max_sentence_num]
            if len(entity_ids) <= self.max_sentence_num:
                entity_ids += [[0] * self.max_seq_len for _ in range(self.max_sentence_num - len(entity_ids))]
           
            assert len(entity_ids) == self.max_sentence_num
            '''
            if len(self.samples[idx]['token_ids']) <= self.max_sentence_num:
                entity_ids = self.samples[idx]['token_ids']
            else:
                #sentence_idx = np.random.choice(len(self.samples[idx]['token_ids']), self.max_sentence_num)
                entity_ids = []
                sentence_idx = []
                for _ in range(self.max_sentence_num):
                    s_idx = np.random.randint(len(self.samples[idx]['token_ids']))
                    while s_idx in sentence_idx:
                        s_idx = np.random.randint(len(self.samples[idx]['token_ids']))
                    sentence_idx.append(s_idx)
                    entity_ids.append(self.samples[idx]['token_ids'][s_idx])
                #print("random_select_sentence: ", self.samples[idx]['title'], sentence_idx, len(self.samples[idx]['token_ids']))
            if len(entity_ids) < self.max_sentence_num:
                entity_ids += [[0] * self.max_seq_len for _ in range(self.max_sentence_num - len(entity_ids))]
            '''
            assert len(entity_ids) == self.max_sentence_num
        return {
            'token_ids': entity_ids,
            'title': self.samples[idx]['title'],
            'title_ids': idx
        }    

def bi_collate_fn(batch):
    token_ids = torch.tensor([sample['token_ids'] for sample in batch]) # sentence_num * max_seq_len
    title = [sample['title'] for sample in batch]
    title_ids = torch.tensor([sample['title_ids'] for sample in batch])
    return {
        'token_ids': token_ids,
        'title': title,
        'title_ids': title_ids
    }

class ZeshelDataset(Dataset):
    def __init__(self, 
        mode, desc_path, dataset_path, tokenizer,
        max_cand_len = 30, max_seq_len = 128, max_sentence_num = 10, all_sentences = False,
    ):
        self.tokenizer = tokenizer
        self.mode = mode
        self.max_cand_len = max_cand_len
        self.max_sentence_num = max_sentence_num
        self.max_seq_len = max_seq_len
        self.all_sentences = all_sentences

        self.entity_desc = {
            world[0]: EncodeDataset(
                document_path = desc_path,
                world = world[0],
                tokenizer = tokenizer,
                max_seq_len = self.max_cand_len,
                max_sentence_num = self.max_sentence_num,
                all_sentences = self.all_sentences
            )
            for world in WORLDS[mode]
        }

        self.load_training_samples(dataset_path, mode, max_seq_len)
        self.subworld_idx = self.get_subworld_idx()

    def get_subworld_idx(self):
        worlds_sample_idx = {world[0]: {'idx': [], 'num': 0} for world in WORLDS[self.mode]}
        for idx, sample in enumerate(self.samples):
            world = sample['world']
            worlds_sample_idx[world]['idx'].append(idx)
            worlds_sample_idx[world]['num'] += 1
        
        return worlds_sample_idx

    def load_training_samples(self, dataset_path, mode, max_seq_len):
        token_path = os.path.join(dataset_path, "{}_token.jsonl".format(mode))
        if os.path.exists(token_path):
            with open(token_path, 'r') as f:
                self.samples = [json.loads(line) for line in f]
                print("Set/{}: Load {} samples".format(mode, len(self.samples)))
        else:
            data_path = os.path.join(dataset_path, "{}.jsonl".format(mode))
            num_lines = sum(1 for line in open(data_path, 'r'))

            self.samples = []
            print("Set/{}: preprocessing {} samples".format(mode, num_lines))
            
            with open(data_path, 'r') as sample_f:
                for sample_line in tqdm(sample_f, total = num_lines):
                    sample = self.tokenize_context(json.loads(sample_line), max_seq_len)
                    self.samples.append(sample)

            with open(token_path, 'w') as f:
                for sample in self.samples:
                    f.write(json.dumps(sample) + '\n')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        context_ids = sample['ids']
        label, world = sample['label'], sample['world']
       
        if self.all_sentences:
            return {
                "label_ids": [-1],
                "context_ids": context_ids,
                "world": world,
                "label_world_idx": label
            }
        else:
            return {
                "label_ids": self.entity_desc[world][label]['token_ids'],
                "context_ids": context_ids,
                "world": world,
                "label_world_idx": label
            }

    def concat_context_entity_ids(self, context_ids, candidate_idx, world):
        if 0 in context_ids:
            context_ids = context_ids[:context_ids.index(0)]

        entity_token_ids = self.entity_desc[world][candidate_idx]['token_ids']
        input_ids = context_ids + entity_token_ids[1:]
        padding = [0] * (self.max_cand_len + self.max_seq_len - len(input_ids))
        input_ids += padding
        assert len(input_ids) == self.max_cand_len + self.max_seq_len

        return input_ids

    def tokenize_context(
        self,
        sample, 
        max_seq_len
    ):
        '''
        https://github.com/facebookresearch/BLINK/blob/master/blink/biencoder/data_process.py
        '''
        mention_tokens = []
        if sample['mention'] and len(sample['mention']) > 0:
            mention_tokens = self.tokenizer.tokenize(sample['mention'])
            mention_tokens = [MENTION_START_TAG] + mention_tokens + [MENTION_END_TAG]

        context_left = sample["context_left"]
        context_right = sample["context_right"]
        context_left = self.tokenizer.tokenize(context_left)
        context_right = self.tokenizer.tokenize(context_right)

        left_quota = (max_seq_len - len(mention_tokens)) // 2 - 1
        right_quota = max_seq_len - len(mention_tokens) - left_quota - 2
        left_add = len(context_left)
        right_add = len(context_right)
        if left_add <= left_quota:
            if right_add > right_quota:
                right_quota += left_quota - left_add
        else:
            if right_add <= right_quota:
                left_quota += right_quota - right_add

        context_tokens = (
            context_left[-left_quota:] + mention_tokens + context_right[:right_quota]
        )

        context_tokens = ["[CLS]"] + context_tokens + ["[SEP]"]
        input_ids = self.tokenizer.convert_tokens_to_ids(context_tokens)
        padding = [0] * (max_seq_len - len(input_ids))
        input_ids += padding
        assert len(input_ids) == max_seq_len

        return {
            "tokens": context_tokens,
            "ids": input_ids,
            "label": sample['label_id'],
            "world": sample['world'],
        }
    
def cross_collate_fn(batch):
    world = [sample['world'] for sample in batch]
    label_world_idx = torch.tensor([sample['label_world_idx'] for sample in batch])
    label_ids = torch.tensor([sample['label_ids'] for sample in batch])
    context_ids = torch.tensor([sample['context_ids'] for sample in batch])
    #label_split = torch.tensor([sample['label_split'] for sample in batch])

    return {
        'context_ids': context_ids,
        'label_ids': label_ids,
        'world': world,
        'label_world_idx': label_world_idx
    }


    