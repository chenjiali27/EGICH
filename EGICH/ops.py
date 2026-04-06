import numpy as np
import torch
import torch.nn.functional as F

def calc_neighbor(label_1, label_2):
    label_1 = label_1.to(torch.float32).cuda()
    label_2 = label_2.to(torch.float32).cuda()
    Sim = (torch.matmul(label_1, label_2.T) > 0).float()
    return Sim

def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

