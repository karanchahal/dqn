import torch
import matplotlib.pyplot as plt
import os
import torch
import torch.distributed as dist
from torch.multiprocessing import Process
import numpy as np
import tracemalloc
import linecache
import os
from scipy import signal

def display_top(snapshot, key_type='lineno', limit=10):
    """
    Debug mem leak
    """
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        print("#%s: %s:%s: %.1f KiB"
              % (index, frame.filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))


""" Gradient averaging. """
def average_gradients(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= size


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def plot_grad_flow(named_parameters):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    print(layers)
    plt.show()

def printshape(fn):
    """
    Utility decorator to print shape of tensor outputting from a func
    """
    def print_shape(*args, **kwargs): 
        output = fn(*args, **kwargs)
        print(f"Shape output by {fn.__name__} is {output.size()}")
        return output  # make sure that the decorator returns the output of fn
    return print_shape 


def get_random_indxs(n, start, end):
    return torch.randint(start, end, (n,))


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.
    input: 
        vector x, 
        [x0, 
         x1, 
         x2]
    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    x = x.numpy()
    return torch.tensor(signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1].copy())


def get_mean_std_across_processes(x):
    size = float(dist.get_world_size())
    summed_all_x = dist.all_reduce(x, op=dist.ReduceOp.SUM)
    x = summed_all_x / size
    return x.mean(), x.std()