from torch import nn


def init_weight(weight, init='normal', init_range=0.1, init_std=0.02):
    if init == 'uniform':
        nn.init.uniform_(weight, -init_range, init_range)
    elif init == 'normal':
        nn.init.normal_(weight, 0.0, init_std)


class SequentialKwargs(nn.Module):
    def __init__(self, *args):
        super(SequentialKwargs, self).__init__()
        self.module_list = nn.ModuleList(args)
        self.module_support_args = [m.forward.__code__.co_varnames for m in self.module_list]

    def forward(self, x, *args, **kwargs):
        for m, supported_args in zip(self.module_list, self.module_support_args):
            module_kwargs = {k: v for k, v in kwargs.items() if k in supported_args}
            x = m(x, *args, **module_kwargs)
        return x