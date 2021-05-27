import torch
import math
import time
import datetime
import random

from torch import Tensor
torch.set_grad_enabled(False)

# Train using SGD Optimizer 
def train_SGD(model, train_input, train_target, learning_rate=0.0001, nb_epochs=500, print_ = False):
    print(f"Training the model on {nb_epochs} epochs with SGD optimizer and learning rate {learning_rate}")
    criterion = LossMSE()
    optimizer = SGD(model.param(), learning_rate)
    
    for epoch in range(nb_epochs):
        
        acc_loss = 0

        for idx, input in enumerate(train_input):
            optimizer.zero_grad()
            output = model.forward(input)
            target = train_target[idx]
            
            loss = criterion.forward(output, target)
            #print(output, target, loss.item())
            acc_loss = acc_loss + loss.item()
            
            grad_loss = criterion.backward()
            model.backward(grad_loss)
            optimizer.step()
            
        if print_ :
            print(f"epoch {epoch}, Loss {acc_loss}")
    return model


# Train using Adam Optimizer
def train_adam(model, train_input, train_target, learning_rate=0.05, nb_epochs=500, print_ = False):
    print(f"Training the model on {nb_epochs} epochs with Adam optimizer and learning rate {learning_rate}")
    criterion = LossMSE()
    optimizer = Adam(model.param(), learning_rate)
    
    for epoch in range(nb_epochs):
        
        acc_loss = 0

        for idx, input in enumerate(train_input):
            optimizer.zero_grad()
            output = model.forward(input)
            target = train_target[idx]
            
            loss = criterion.forward(output, target)
            #print(output, target, loss.item())
            acc_loss = acc_loss + loss.item()
            
            grad_loss = criterion.backward()
            model.backward(grad_loss)
            optimizer.step()
        
        if print_:
            print(f"epoch {epoch}, Loss {acc_loss}")
    return model

# Accuracy and plot if the output is only one node 

def accuracy_and_plot_one_output(model, test_input, test_label):
    nb_correct = 0
    nb_total = test_input.shape[0]
    zeros = []
    ones = []
    
    for idx, input in enumerate(test_input):
        prediction = model.forward(input)
        target = test_label[idx]
        if prediction.round() == 0 :
            zeros.append(input)
        else :
            ones.append(input)
        
        nb_correct += prediction.round() == target
    zeros = torch.stack(zeros)
    ones = torch.stack(ones)
    plt.figure(figsize= (6,6))
    plt.scatter(zeros[:,0], zeros[:,1], label='zeros')
    plt.scatter(ones[:,0], ones[:,1], label= 'ones')
    plt.legend()
    plt.show()
    accuracy = 100 * nb_correct.item() / nb_total
    error_rate = 100 - accuracy
    print(f'Accuracy is : {accuracy}%')
    print(f'Error rate : {error_rate}%')

def accuracy_one_output(model, test_input, test_label, print_ = True):
    nb_correct = 0
    nb_total = test_input.shape[0]
    
    for idx, input in enumerate(test_input):
        prediction = model.forward(input)
        target = test_label[idx]
        
        nb_correct += prediction.round() == target
    accuracy = 100 * nb_correct.item() / nb_total
    error_rate = 100 - accuracy
    if print_ :
        print(f'Accuracy is : {accuracy}%')
        print(f'Error rate : {error_rate}%')
    
    return accuracy, error_rate

# Accuracy and Plot if there are two output nodes 
# Compute the accuracy and plot the results 
def accuracy_and_plot_two_output(model, test_input, test_label):
    nb_correct = 0
    nb_total = test_input.shape[0]
    zeros = []
    ones = []
    
    for idx, input in enumerate(test_input):
        prediction = model.forward(input)
        target = test_label[idx]
        if prediction.argmax() == 0 :
            zeros.append(input)
        else :
            ones.append(input)
        
        nb_correct += prediction.argmax() == target
    zeros = torch.stack(zeros)
    ones = torch.stack(ones)
    plt.figure(figsize= (6,6))
    plt.scatter(zeros[:,0], zeros[:,1], label='zeros')
    plt.scatter(ones[:,0], ones[:,1], label= 'ones')
    plt.legend()
    plt.show()
    accuracy = 100 * nb_correct.item() / nb_total
    error_rate = 100 - accuracy
    print(f'Accuracy is : {accuracy}%')
    print(f'Error rate : {error_rate}%')
    
    return accuracy, error_rate

def accuracy_two_output(model, test_input, test_label, print_ = True):
    nb_correct = 0
    nb_total = test_input.shape[0]
    
    for idx, input in enumerate(test_input):
        prediction = model.forward(input)
        target = test_label[idx]
        
        nb_correct += prediction.argmax() == target
    accuracy = 100 * nb_correct.item() / nb_total
    error_rate = 100 - accuracy
    if print_ :
        print(f'Accuracy is : {accuracy}%')
        print(f'Error rate : {error_rate}%')
    
    return accuracy, error_rate

# Train the model and plot the curve of error rate by epoch 

def train_and_plot_accuracy(model, train_input, train_target_one_hot, train_target, test_input, test_target, optim="sgd", learning_rate=0.0001, nb_epochs=500, one_output=False, print_=False):
    print(f"Training the model on {nb_epochs} epochs with {optim} optimizer and learning rate {learning_rate}")
    criterion = LossMSE()
    if optim == "adam":
        optimizer = Adam(model.param(), learning_rate)
    else :
        optimizer = SGD(model.param(), learning_rate)
    
    test_errs = []
    train_errs = []
    for epoch in range(nb_epochs):
        
        acc_loss = 0

        for idx, input in enumerate(train_input):
            optimizer.zero_grad()
            output = model.forward(input)
            target = train_target_one_hot[idx]
            
            loss = criterion.forward(output, target)
            #print(output, target, loss.item())
            acc_loss = acc_loss + loss.item()
            
            grad_loss = criterion.backward()
            model.backward(grad_loss)
            optimizer.step()
        
        if one_output:
            _, test_err = accuracy_one_output(model, test_input, test_target, print_ = False)
            _, train_err = accuracy_one_output(model, train_input, train_target, print_ = False)
        else : 
            _, test_err = accuracy_two_output(model, test_input, test_target, print_ = False)
            _, train_err = accuracy_two_output(model, train_input, train_target, print_ = False)
            
        test_errs.append(test_err)
        train_errs.append(train_err)
        
        avg_acc_loss = acc_loss / train_input.shape[0]
        if print_:
            print(f"epoch {epoch}, Loss {avg_acc_loss}")
    plt.figure(figsize= (10,10))
    plt.plot(test_errs, label = "Test Error Rate")
    plt.plot(train_errs, label = "Train Error Rate")
    plt.legend()
    plt.savefig(f"Test_and_train_loss_{optim}_.png")
    plt.show()
    return model