import torch
import math
import time
import datetime
import random

from torch import Tensor
torch.set_grad_enabled(False)




# Plotting functions 

def plot_from_input(input, target):
    zeros = input[(target == 0).nonzero()[:,0]]
    ones = input[target.nonzero()[:,0]]
    plt.figure(figsize= (6,6))
    plt.scatter(zeros[:,0], zeros[:,1], label='zeros')
    plt.scatter(ones[:,0], ones[:,1], label= 'ones')
    plt.legend()
    plt.savefig('generated_data.png')
    plt.show()