import torch
import matplotlib.pyplot as plt

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

