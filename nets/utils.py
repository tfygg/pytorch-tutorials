import torch
import torch.nn as nn
import torch.nn.init as init

def initNetParams(net):
    '''Init net parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.xavier_uniform(m.weight)
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)

"""
w = torch.Tensor(3, 5)
print(w)
print("uniform :", init.uniform(w))
print("normal :", init.normal(w))
print("xavier_uniform :", init.xavier_uniform(w))
print("xavier_normal :", init.xavier_normal(w))
print("kaiming_uniform :", init.kaiming_uniform(w))
print("kaiming_normal :", init.kaiming_normal(w))
"""

