import matplotlib.pyplot as plt 
import torch
def mp(ls):
    return list(map(lambda x : x / 10,ls))

# Plots the average error rate for each trial 
def plot_results(base, ws, ws_al, dropout):
    plt.plot(mp(base), label='base')
    plt.plot(mp(ws), label='ws')
    plt.plot(mp(ws_al), label='ws + al')
    plt.plot(mp(dropout), label='Dropout')
    plt.xlabel("Trial Number")
    plt.ylabel("Test Error Rate (%)")
    plt.legend()
    plt.savefig("benchmark_results.png")
    plt.show()

    
# Compute the number of parameters for a given model 
def compute_nb_parameters(model, name = "Model"):
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    pytorch_total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f'{name} has {pytorch_total_params} parameters. {pytorch_total_trainable_params} of them are trainable')
    