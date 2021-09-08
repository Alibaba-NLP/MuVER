#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   evaluate.py
@Time    :   2021/03/17 23:41:28
@Author  :   Xinyin Ma
@Version :   0.1
@Contact :   maxinyin@zju.edu.cn
'''
import time 
import random

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.distributed as dist

from prettytable import PrettyTable     
from muver.utils.params import WORLDS
from data_loader import bi_collate_fn, cross_collate_fn
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import SequentialSampler


def pretty_visualize(scores, top_k):
    rows = []
    for world, score in scores.items():
        rows.append([world] + [round(s * 1.0 / score[1], 4) for s in score[0]])
    
    table = PrettyTable()
    table.field_names = ["World"] + ["R@{}".format(k) for k in top_k]
    table.add_rows(rows)
    print(table)

def evaluate_bi_model(model, tokenizer, dataset, mode, encode_batch_size = 16, device = "cpu", local_rank = -1, n_gpu = 1, 
                      view_expansion = False, top_k = 0.4, merge_layers = 3, is_accumulate_score = False):
    
    model.eval()
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        test_module = model.module
    else:
        test_module = model

    world_entity_pool, world_entity_titles = {}, {}
    for world, world_dataset in dataset.entity_desc.items():
        entity_pool, entity_title = [], []
        if n_gpu > 1:
            sampler = DistributedSampler(world_dataset)
        else:
            sampler = SequentialSampler(world_dataset)
        encode_dataloader = DataLoader(dataset = world_dataset, batch_size = 1, collate_fn=bi_collate_fn, shuffle=False, sampler=sampler)
        
        disable = True if local_rank not in [-1, 0] else False
        for sample in tqdm(encode_dataloader, disable=disable):
            candidate_encode = test_module.encode_candidates(
                ent_ids = sample['token_ids'].cuda(),
                view_expansion = view_expansion,
                top_k = top_k, 
                merge_layers = merge_layers,
                mode='test'
            ).squeeze(0).detach().to("cpu") # not support for encode_batch_size > 1
            entity_pool.append(candidate_encode)
            entity_title += [sample['title'][0]] * candidate_encode.size(0)
            
        world_entity_pool[world] = torch.cat(entity_pool, 0)
        world_entity_titles[world] = entity_title
    
    torch.save([world_entity_pool, world_entity_titles], 'entity_{}.pt'.format(local_rank))
    

    if n_gpu > 1:
        torch.distributed.barrier()
    
    if local_rank not in [-1, 0]:
        return None, None

    world_entity_pool, world_entity_titles = {}, {}
    for i in range(n_gpu):
        sub_entity_pool, sub_entity_titles = torch.load('entity_{}.pt'.format(i), map_location='cpu') 
        for world_name, world_num in WORLDS[mode]:
            titles = world_entity_titles.get(world_name, [])
            pool = world_entity_pool.get(world_name, [])

            sub_titles = sub_entity_titles[world_name]
            sub_pool = sub_entity_pool[world_name]
            if world_num % n_gpu and world_num % n_gpu - 1 < i:
                end_idx = len(sub_titles) - 2
                while sub_titles[end_idx] == sub_titles[-1]:
                    end_idx -= 1
                
                sub_titles = sub_titles[:end_idx + 1]
                sub_pool = sub_pool[:end_idx+1, :]

            titles += sub_titles
            world_entity_titles[world_name] = titles

            pool.append(sub_pool)
            world_entity_pool[world_name] = pool
    
    for key, _ in WORLDS[mode]:
        pool = world_entity_pool[key]
        pool = torch.cat(pool, 0).to("cuda:0")
        world_entity_pool[key] = pool 
        #print(world_entity_pool[key].shape)

    world_entity_ids_range = {}
    for key, titles in world_entity_titles.items():
        ids_range = {}
        for ids, title in enumerate(titles):
            title_range = ids_range.get(title, [])
            title_range.append(ids)
            ids_range[title] = title_range
        world_entity_ids_range[key] = ids_range

    top_k = [1, 2, 4, 8, 16, 32, 50, 64]
    score_metrics = {world_name: [[0] * len(top_k), 0] for world_name, _ in WORLDS[mode]}
    score_metrics['total'] = [[0] * len(top_k), 0]
    candidates = []
    # Then Encode the entities and Compare
    dataloader = DataLoader(dataset=dataset, batch_size=encode_batch_size, collate_fn=cross_collate_fn, shuffle=False)
    for batch in tqdm(dataloader):
        worlds, labels = batch['world'], batch['label_world_idx']
        predict_scores = test_module.score_candidates(
            ctx_ids = batch['context_ids'].to("cuda:0"), 
            ctx_world = batch['world'],
            candidate_pool = world_entity_pool
        ) # [candidates_num] * batch_size
         
        for predict_score, world, label in zip(predict_scores, worlds, labels):
            predict_score = torch.softmax(predict_score, -1)
            predict_ids = torch.sort(predict_score, -1, descending=True).indices.cpu()
            scores = torch.sort(predict_score, -1, descending=True).values.cpu()
            label_title = dataset.entity_desc[world].get_nth_title(label)
            accumulate_score = is_accumulate_score
            if accumulate_score:
                predict_title_dict = {}

                ids = 0
                while len(predict_title_dict.keys()) < 200:#top_k[-1]:
                    title = world_entity_titles[world][predict_ids[ids]] 
                    title_score = predict_title_dict.get(title, 0) + scores[ids]
                    predict_title_dict[title] = title_score
                    
                    ids += 1
                predict_title = sorted(predict_title_dict, key=predict_title_dict.get)[::-1]
            else:
                predict_title = []
                ids = 0
                while len(predict_title) < 64:
                    title = world_entity_titles[world][predict_ids[ids]]
                    if title not in predict_title:
                        predict_title.append(title)
                    ids += 1

            for k_idx, k in enumerate(top_k):
                if label_title in predict_title[:k]:
                    score_metrics[world][0][k_idx] += 1
                    score_metrics['total'][0][k_idx] += 1
            score_metrics[world][1] += 1
            score_metrics['total'][1] += 1
        
            candidates.append([{'title': title} for title in predict_title])
    print(score_metrics)
    pretty_visualize(score_metrics, top_k)

    return score_metrics['total'][0][-1], candidates

def evaluate_cross_model(model, tokenizer, dataset, mode, encode_batch_size = 1, device = "cpu"):
    model.eval()
    if isinstance(model, torch.nn.DataParallel):
        test_module = model.module
    else:
        test_module = model

    dataloader = DataLoader(dataset=dataset, batch_size=encode_batch_size, num_workers=1, shuffle=False, collate_fn=collate_fn)
    #normalized_correct, normalized_total, unnormalized_total = 0, 0, 0
    score_metrics = {world_name: [0, 0, 0] for world_name, _ in WORLDS[mode]}
    score_metrics['total'] = [0, 0, 0]
    for batch in tqdm(dataloader):
        ctx_ids = batch['candidate_ids'].to(device)
        split_len = batch['split_len']
        if encode_batch_size == 1:
            ctx_ids = ctx_ids.squeeze(0)
            split_len = split_len[0]

        score, _ = model(ctx_ids, split_len=split_len, target = None) # batch_size * top_k
        predict_idx = torch.max(score, -1).indices

        for idx, t, w in zip(predict_idx, batch['label'], batch['world']):
            if idx == t:
                score_metrics[w][0] += 1
                score_metrics['total'][0] += 1
            if t != -1:
                score_metrics[w][1] += 1
                score_metrics['total'][1] += 1

            score_metrics[w][2] += 1
            score_metrics['total'][2] += 1

    return score_metrics
        
        
    


            

        

