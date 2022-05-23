#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   train.py
@Time    :   2021/09/08 10:36:15
@Author  :   Xinyin Ma
@Version :   1.0
@Contact :   maxinyin@zju.edu.cn
'''

import os
import time
import json
import random
import argparse
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, RandomSampler

from transformers import BertTokenizerFast, AdamW, get_linear_schedule_with_warmup

from muver.utils.logger import LoggerWithDepth
from muver.utils.tools import grid_search_hyperparamters, set_random_seed

from data_loader import EncodeDataset, ZeshelDataset, cross_collate_fn, SubworldBatchSampler, SubWorldDistributedSampler
from model import BiEncoder, NCE_Random
from zeshel_evaluate import evaluate_bi_model



def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_model', type=str, required=True)
    parser.add_argument('--dataset_path', type=str, default='data/zeshel')
    parser.add_argument('--bi_ckpt_path', type=str, nargs='+', default=None)

    parser.add_argument('--epoch', type=int, default=5)
    parser.add_argument('--train_batch_size', type=int, default=12)
    parser.add_argument('--max_cand_len', type=int, default=40)
    parser.add_argument('--max_seq_len', type=int, default=128)
    parser.add_argument('--max_sentence_num', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, nargs='+', default=1e-5)
    parser.add_argument('--weight_decay', type=float, nargs='+', default=0.01)
    parser.add_argument('--warmup_ratio', type=float, nargs='+', default=0.1)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--gradient_accumulation', type=int, default=1)
    parser.add_argument('--merge_layers', type=int, default=3)
    parser.add_argument('--top_k', type=float, default=0.4)
    #parser.add_argument('--alpha', type=float, default=0.5)
    #parser.add_argument('--beta', type=float, nargs='+', default=50)

    parser.add_argument('--eval_batch_size', type=int, default=8)
    parser.add_argument('--logging_interval', type=int, default=50)
    parser.add_argument('--eval_interval', type=int, default=2000)
    parser.add_argument('--accumulate_score', action="store_true")

    parser.add_argument('--do_train', action="store_true")
    parser.add_argument('--do_eval', action="store_true")
    parser.add_argument('--do_test', action="store_true")
    parser.add_argument('--view_expansion', action="store_true")
    parser.add_argument('--test_mode', type=str, default='test')

    parser.add_argument("--data_parallel", action="store_true")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--name", type=str, default='test')
    return parser.parse_args()


def main(local_rank, args, train_dataset, valid_dataset, test_dataset, tokenizer):
    args.local_rank = local_rank
    if args.do_train and args.local_rank in [0, -1]:
        logger = LoggerWithDepth(
            env_name=args.name, 
            config=args.__dict__,
        )
    else:
        logger = None

    # Set Training Device
    if args.data_parallel:
        
        if args.n_gpu == 1:
            args.data_parallel = False
        else:    
            dist.init_process_group("nccl", rank=args.local_rank, world_size=args.n_gpu)
            torch.cuda.set_device(args.local_rank)

    args.device = "cuda" if not args.no_cuda else "cpu"
    set_random_seed(args.seed)

    grid_arguments = grid_search_hyperparamters(args)
    for grid_args in grid_arguments:
        if args.do_train and args.local_rank in [0, -1]:
            sub_name = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
            logger.setup_sublogger(sub_name, grid_args.__dict__)
            
        # Load Model and Tokenizer
        bi_model = BiEncoder(args.pretrained_model).cuda()
        
        criterion = NCE_Random(args.n_gpu)

        # Load From checkpoint
        if grid_args.bi_ckpt_path is not None:
            state_dict = torch.load(grid_args.bi_ckpt_path, map_location='cpu')
            new_state_dict = {}
            for param_name, param_value in state_dict.items():
                if param_name[:7] == 'module.':
                    new_state_dict[param_name[7:]] = param_value
                else:
                    new_state_dict[param_name] = param_value
            bi_model.load_state_dict(new_state_dict)

        if args.n_gpu > 1:
            bi_model = DDP(bi_model, device_ids=[args.local_rank], find_unused_parameters=True)
        
        # Load Data
        if args.do_train:
            train_batch_size = grid_args.train_batch_size // grid_args.gradient_accumulation
            
            if args.data_parallel:
                sampler = SubWorldDistributedSampler(batch_size=grid_args.train_batch_size, subworld_idx=train_dataset.subworld_idx, num_replicas=args.n_gpu, rank=args.local_rank)
            else:
                sampler = SubworldBatchSampler(batch_size=grid_args.train_batch_size, subworld_idx=train_dataset.subworld_idx)
            train_dataloader = DataLoader(dataset=train_dataset, batch_sampler=sampler, collate_fn = cross_collate_fn)

             # optimizer & scheduler
            no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {'params': [p for n, p in bi_model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': grid_args.weight_decay},
                {'params': [p for n, p in bi_model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
            ]
            optimizer = AdamW(optimizer_grouped_parameters, lr=grid_args.learning_rate)

            total_steps = len(train_dataset) * args.epoch // train_batch_size
            warmup_steps = int(grid_args.warmup_ratio * total_steps)
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
            if args.local_rank in [0, -1]:
                logger.writer.info("Optimization steps = {},  Warmup steps = {}".format(total_steps, warmup_steps))

        # Train
        if args.do_train:
            step, max_score = 0, 0
            with tqdm(total = total_steps) as pbar:
                for e in range(args.epoch):
                    tr_loss = []
                    for batch in train_dataloader:
                        bi_model.train()
                        step += 1

                        world = batch['world']
                        for w in world[1:]:
                            assert world[0] == w
                        
                        ctx_ids, ent_ids = batch['context_ids'], batch['label_ids']
                        
                        ctx_ids = ctx_ids.cuda(non_blocking=True)
                        ent_ids = ent_ids.cuda(non_blocking=True)
                        ctx_output, ent_output = bi_model(
                            ctx_ids = ctx_ids, 
                            ent_ids = ent_ids, 
                            num_gpus = args.n_gpu
                        )
                        loss, bi_acc, bi_score = criterion(ctx_output, ent_output)
                        loss.backward()
                        
                        #if args.n_gpu > 1:
                        #    dist.all_reduce(loss.div_(args.n_gpu))
                        if step % grid_args.gradient_accumulation == 0:
                            if args.max_grad_norm > 0:
                                torch.nn.utils.clip_grad_norm_(bi_model.parameters(), args.max_grad_norm)
                            optimizer.step()
                            scheduler.step()
                            bi_model.zero_grad()
                        
                        if args.local_rank in [0, -1]:
                            pbar.set_description("epoch: {}, loss: {}, acc: {}".format(
                                e + 1, loss.item(), bi_acc.item()
                            ))  
                            pbar.update()

                        tr_loss.append(loss.item())
                        if step % args.logging_interval == 0 and args.local_rank in [0, -1]:
                            logger.writer.info("Step {}: Average Loss = {}".format(step, sum(tr_loss) / len(tr_loss)))
                            tr_loss = []
                        
                        if step % args.eval_interval == 0 and args.do_eval:
                            with torch.no_grad():
                                score, _ = evaluate_bi_model(
                                    bi_model, tokenizer, valid_dataset, 
                                    mode='valid', 
                                    device=args.device, 
                                    local_rank=args.local_rank, 
                                    n_gpu=args.n_gpu)
                                
                                if args.local_rank in [0, -1]: 
                                    logger.writer.info(score)
                                    torch.save(bi_model.state_dict(), logger.lastest_checkpoint_path)

                                    if max_score < score:
                                        torch.save(bi_model.state_dict(), logger.checkpoint_path)
                                        max_score = score
                                
                            
                    if args.local_rank in [0, -1]:
                        torch.save(bi_model.state_dict(), os.path.join(logger.sub_dir, 'epoch_{}.bin'.format(e)))

            grid_args.best_evaluation_score = max_score
            if args.local_rank in [0, -1]:
                logger.write_description_to_folder(os.path.join(logger.sub_dir, 'description.txt'), grid_args.__dict__)  
            del optimizer
            
        if args.do_test:
            if args.do_train and args.local_rank in [0, -1]:
                bi_model.load_state_dict(torch.load(logger.checkpoint_path, map_location='cpu'))
            with torch.no_grad():
                score, candidates = evaluate_bi_model(
                    bi_model, tokenizer, test_dataset, 
                    mode=args.test_mode, 
                    device = args.device, 
                    local_rank=args.local_rank,
                    n_gpu=args.n_gpu,
                    encode_batch_size=args.eval_batch_size,
                    view_expansion = args.view_expansion,
                    is_accumulate_score = args.accumulate_score,
                    merge_layers=args.merge_layers,
                    top_k=args.top_k)

            if args.local_rank in [-1, 0]:
                if logger is not None:
                    result_path = os.path.join(logger.sub_dir, 'score.json')
                    candidate_path = os.path.join(logger.sub_dir, 'candidates.json')
                else:
                    dir_path = os.path.dirname(os.path.abspath(grid_args.bi_ckpt_path))
                    result_path = os.path.join(dir_path,  '{}_score.json'.format(args.test_mode))
                    candidate_path = os.path.join(dir_path, '{}_candidates.json'.format(args.test_mode))

                with open(result_path, 'w') as f:
                    f.write(json.dumps(score))
                
                with open(candidate_path, 'w') as f:
                    for candidate in candidates:
                        f.write(json.dumps(candidate) + '\n')
         
        del bi_model

if __name__ == "__main__":
    args = argument_parser()
    print(args.__dict__)

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "18101"

    args.n_gpu = torch.cuda.device_count()
    
    # before multiprocessing, preprocess the data
    train_dataset, valid_dataset, test_dataset = None, None, None
    tokenizer = BertTokenizerFast.from_pretrained(args.pretrained_model)
    
    if args.do_train:
        train_dataset = ZeshelDataset(
            mode='train',
            desc_path=os.path.join(args.dataset_path, 'documents'),
            dataset_path=os.path.join(args.dataset_path, 'blink_format'),
            tokenizer=tokenizer,
            max_cand_len=args.max_cand_len,
            max_sentence_num=args.max_sentence_num,
            max_seq_len=args.max_seq_len,
        )

    if args.do_eval:
        valid_dataset = ZeshelDataset(
            mode='valid',
            desc_path=os.path.join(args.dataset_path, 'documents'),
            dataset_path=os.path.join(args.dataset_path, 'blink_format'),
            tokenizer=tokenizer,
            max_cand_len=args.max_cand_len,
            max_seq_len=args.max_seq_len,
            all_sentences = True
        )

    if args.do_test:
        test_dataset = ZeshelDataset(
            mode=args.test_mode,
            desc_path=os.path.join(args.dataset_path, 'documents'),
            dataset_path=os.path.join(args.dataset_path, 'blink_format'),
            tokenizer=tokenizer,
            max_cand_len=args.max_cand_len,
            max_seq_len=args.max_seq_len,
            all_sentences = True
        )
    
    mp.spawn(main, args=(args, train_dataset, valid_dataset, test_dataset, tokenizer,), nprocs=args.n_gpu, join=True)
