from torch.utils.checkpoint import checkpoint_sequential
import torch
import pandas as pd
import torch.nn as nn
from torchvision.models import resnet18
import matplotlib.pyplot as plt 

def _get_gpu_mem(synchronize=True, empty_cache=True):
    return torch.cuda.memory_allocated(), torch.cuda.memory_cached()


def _generate_mem_hook(handle_ref, mem, idx, hook_type, exp):
    def hook(self, *args):
        
        print('hook called')

        if len(mem) == 0:
            call_idx = 0
        else:
            call_idx = mem[-1]["call_idx"] + 1

        mem_all, mem_cached = _get_gpu_mem()
        torch.cuda.synchronize()
        mem.append({
            'layer_idx': idx,
            'call_idx': call_idx,
            'layer_type': type(self).__name__,
            'exp': exp,
            'hook_type': hook_type,
            'mem_all': mem_all,
            'mem_cached': mem_cached,
        })

        if exp=='checkpointed':
            print(m[-1])

    return hook


def _add_memory_hooks(idx, mod, mem_log, exp, hr):
    h = mod.register_forward_pre_hook(_generate_mem_hook(hr, mem_log, idx, 'pre', exp))
    hr.append(h)

    h = mod.register_forward_hook(_generate_mem_hook(hr, mem_log, idx, 'fwd', exp))
    hr.append(h)

    h = mod.register_backward_hook(_generate_mem_hook(hr, mem_log, idx, 'bwd', exp))
    hr.append(h)

    print('Hooks added for {}'.format(exp))


def log_mem(model, inp, exp, cp_chunks=1):
    mem_log = []
    hr = []
    for idx, module in enumerate(model.modules()):
        _add_memory_hooks(idx, module, mem_log, exp, hr)

    try:
        if cp_chunks > 1:
            out = checkpoint_sequential(model, cp_chunks, inp)
        else:
            out = model(inp)
        loss = out.sum()
        loss.backward()
    finally:
        [h.remove() for h in hr]

        return mem_log

def plot_mem(df, exps=None, normalize_call_idx=True, normalize_mem_all=True):

    if exps is None:
        exps = df.exp.drop_duplicates()

    fig, ax = plt.subplots(figsize=(20, 10))
    
    for exp in exps:
        try:
            df_ = df[df.exp == exp]
        except AttributeError:
            continue

        print(df_)

        if normalize_call_idx:
            df_.call_idx = df_.call_idx / df_.call_idx.max()

        if normalize_mem_all:
            df_.mem_all = df_.mem_all - df_[df_.call_idx == df_.call_idx.min()].mem_all.iloc[0]
            df_.mem_all = df_.mem_all // 2 ** 20

        plot = df_.plot(ax=ax, x='call_idx', y='mem_all', label=exp)

        plot.get_figure().savefig('{}'.format(exp))


if __name__=='__main__':
    model = resnet18().cuda()
    bs = 128
    input = torch.rand(bs, 3, 224, 224).cuda()
    mem_log = []

    mem_log.extend(log_mem(model, input, exp='baseline'))
    mem_log.extend(log_mem(model, input, exp='checkpointed', cp_chunks=3))

    df = pd.DataFrame(mem_log)
    plot_mem(df, exps=['baseline','checkpointed'])
