import torch
import numpy as np
import time


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def get_exp_time():
    return time.strftime("ClusterDrop_%y_%m_%d-%H-%M-%S", time.localtime())
