import os.path as osp
import sys
import time
from easydict import EasyDict as edict

import numpy as np

C = edict()
config = C
cfg = C

"""please config ROOT_dir and user when u first using"""
C.root_dir = osp.realpath(".")

exp_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())

"""Opflow Model Config"""
C.opflow_model_path = 'PATH/TO/YOUR/raft-things.pth'
C.small = False
C.mixed_precision = False
C.alternate_corr = False
C.opflow_num_iters = 10

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

add_path(osp.join(C.root_dir, 'core'))
