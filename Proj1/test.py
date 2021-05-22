import torch
import math
import dlc_practical_prologue as prologue
import models as m
import train_functions as tr
import utilities as u
from torch import optim
from torch import Tensor
from torch import nn
from torch.nn import functional as F


# Function to benchmark a given model 

def benchmark_model(model, train_function, evaluate_function, nb_trials=20, N=1000, mini_batch_size=250, nb_epochs=25, model_requires_target_and_classes=False, _print=False):
    # Benchmark of the basic network with Adam optimizer
    performances = []
    for trial in range(nb_trials):

        # Generate Data 
        train_input, train_target, train_classes, test_input, test_target, test_classes = prologue.generate_pair_sets(N)
        test_target_one_hot = prologue.convert_to_one_hot_labels(test_input, test_target)

        # Define the model 
        model_total = model()

        # Train the model
        if model_requires_target_and_classes : 
            train_function(model_total, train_input, train_target, train_classes, mini_batch_size=mini_batch_size,
                           nb_epochs=nb_epochs, use_optimizer="adam", _print=_print)
        else :
            train_function(model_total, train_input, train_target, mini_batch_size=mini_batch_size,
                           nb_epochs=nb_epochs, use_optimizer="adam", _print=_print)

        # Evaluate performances 
        nb_test_errors = evaluate_function(model_total, test_input, test_target_one_hot, mini_batch_size=mini_batch_size)
        print('test error Net trial {:d} {:0.2f}% {:d}/{:d}'.format(trial, (100 * nb_test_errors) / test_input.size(0),
                                                              nb_test_errors, test_input.size(0)))
        performances.append(nb_test_errors)

    mean_perf = 100 * sum(performances) / (N * nb_trials)
    print(f"Average precision of this architecture {mean_perf}%")
    
    std_dev = math.sqrt(sum(list(map(lambda x : x - mean_perf,performances))))/nb_trials
    print(f"With standard deviation of  {std_dev}")
    
    return performances
    

print("Benchmark of the baseline model")
results_base = benchmark_model(m.Base_Net, tr.train_model_base_ws, tr.compute_nb_errors_base_ws)

print("Benchmark of the model with Weight Sharing")
results_ws = benchmark_model(m.Weight_Sharing_Net, tr.train_model_base_ws, tr.compute_nb_errors_base_ws)

print("Benchmark of the model with Weight Sharing and an auxiliary loss")
results_ws_al = benchmark_model(m.Auxiliary_Loss_Weight_Sharing_Net, tr.train_model_auxiliary_loss, tr.compute_nb_errors_auxilary_loss, model_requires_target_and_classes=True)

print("Benchmark of the model with Weight Sharing and an auxiliary loss and with additionnal Dropout layers (50 epochs)")
results_dropout = benchmark_model(m.Auxiliary_Loss_Net_Dropout, tr.train_model_auxiliary_loss, tr.compute_nb_errors_auxilary_loss, model_requires_target_and_classes=True, nb_epochs=50)

u.plot_results(results_base, results_ws, results_ws_al, results_dropout)