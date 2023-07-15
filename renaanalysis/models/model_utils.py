from torch import nn


def init_weight(weight, init='normal', init_range=0.1, init_std=0.02):
    if init == 'uniform':
        nn.init.uniform_(weight, -init_range, init_range)
    elif init == 'normal':
        nn.init.normal_(weight, 0.0, init_std)