import torch
import math
from torch import optim
from torch import Tensor
from torch import nn
from torch.nn import functional as F

# Training function for both Baseline and Weight Sharing model 
def train_model_base_ws(model, train_input, train_target, mini_batch_size, nb_epochs = 100, use_optimizer= None, _print=False):
    criterion = nn.CrossEntropyLoss()
    eta = 1e-3
    if use_optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=eta)
    if use_optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=eta)
    for e in range(nb_epochs):
        acc_loss = 0

        for b in range(0, train_input.size(0), mini_batch_size):
            output = model(train_input.narrow(0, b, mini_batch_size))
            target = train_target.narrow(0, b, mini_batch_size).long()
            loss = criterion(output, target)
            acc_loss = acc_loss + loss.item()
 
            model.zero_grad()
            loss.backward()
            
            if use_optimizer != None :
                optimizer.step()
            else :
                with torch.no_grad():
                    for p in model.parameters():
                        p -= eta * p.grad
        if _print:
            print(e, acc_loss)

# Error evaluation function for both baseline model and weight sharing 
def compute_nb_errors_base_ws(model, input, target, mini_batch_size):
    nb_errors = 0

    for b in range(0, input.size(0), mini_batch_size):
        output = model(input.narrow(0, b, mini_batch_size))
        _, predicted_classes = output.max(1)
        for k in range(mini_batch_size):
            if target[b + k, predicted_classes[k]] <= 0:
                nb_errors = nb_errors + 1

    return nb_errors

# Training function for the auxiliary loss with weight sharing model 
def train_model_auxiliary_loss(model, train_input, train_target, train_classes, mini_batch_size, nb_epochs = 100, use_optimizer= None, _print=False):
    criterion_auxilary = nn.CrossEntropyLoss()
    criterion_final = nn.CrossEntropyLoss()
    
    eta = 1e-3
    if use_optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=eta)
    if use_optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=eta)
    for e in range(nb_epochs):
        acc_loss = 0

        for b in range(0, train_input.size(0), mini_batch_size):
            digit_1, digit_2, comparison = model(train_input.narrow(0, b, mini_batch_size))
            
            target_comparison = train_target.narrow(0, b, mini_batch_size).long()
            
            target_digit_1, target_digit_2 = train_classes.narrow(0, b, mini_batch_size)[:,0], train_classes.narrow(0, b, mini_batch_size)[:,1]
            loss1 = criterion_auxilary(digit_1, target_digit_1)
            loss2 = criterion_auxilary(digit_2, target_digit_2)
            loss3 = criterion_final(comparison, target_comparison)
            acc_loss = acc_loss + loss1.item() + loss2.item() + loss3.item()
 
            model.zero_grad()
            loss1.backward(retain_graph=True)
            loss2.backward(retain_graph=True)
            loss3.backward()
            
            if use_optimizer != None :
                optimizer.step()
            else :
                with torch.no_grad():
                    for p in model.parameters():
                        p -= eta * p.grad
        if _print :
            print(e, acc_loss)

# Error evaluation function for the Auxiliary Loss with weight sharing model       
def compute_nb_errors_auxilary_loss(model, input, target, mini_batch_size):
    nb_errors = 0

    for b in range(0, input.size(0), mini_batch_size):
        _, _, output = model(input.narrow(0, b, mini_batch_size))
        _, predicted_classes = output.max(1)
        for k in range(mini_batch_size):
            if target[b + k, predicted_classes[k]] <= 0:
                nb_errors = nb_errors + 1

    return nb_errors