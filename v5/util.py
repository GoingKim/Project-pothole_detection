import os
import math
import time
import random

import shutil
import numpy as np
import pandas as pd

import torch

from datetime import datetime


##
def init_logger(path):
    from logging import getLogger, INFO, FileHandler,  Formatter,  StreamHandler

    if not os.path.exists(path):
        os.makedirs(path)

    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=os.path.join(path, 'train_log'))
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger


##
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0    # val is value
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))


##
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


## save model
def save_model(ckpt_dir, net, fold, num_epoch, epoch, batch, save_argument):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    suffix = 'f{}_ep{}_bt{}_date{}.pth'.format(fold, epoch, batch, datetime.now().strftime("%m.%d-%H:%M"))

    filename = os.path.join(ckpt_dir, 'checkpoint_0.pth')
    best_filename = os.path.join(ckpt_dir, 'best_checkpoint_0.pth')
    final_filename = os.path.join(ckpt_dir, 'final_' + suffix)

    # save model every epoch
    # If you want to save model every epoch, change filename to suffix
    torch.save(net.state_dict(), filename)

    # leave only best model
    if save_argument:
        shutil.copyfile(filename, best_filename)

    if num_epoch == epoch:
        shutil.copyfile(best_filename, final_filename)
        os.remove(filename)
        os.remove(best_filename)