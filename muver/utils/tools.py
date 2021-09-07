import os
import random
from copy import deepcopy
from itertools import product

import numpy as np
import torch

def grid_search_hyperparamters(args):
    search_key, search_value = [], []
    for key, value in args.__dict__.items():
        if isinstance(value, list) and key != 'test_set':
            search_key.append(key)
            search_value.append(value)
    
    new_args = []
    for one_search_value in product(*search_value):
        arg = deepcopy(args)
        for key, value in zip(search_key, one_search_value):
            arg.__setattr__(key, value)
        new_args.append(arg)
    return new_args

def set_random_seed(seed):
    # Set Random Seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)