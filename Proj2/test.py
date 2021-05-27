import torch
import math
import time
import datetime
import modules
import generate_data
import train_and_evaluate

from modules import * 
from generate_data import * 
from train_and_evaluate import * 

from torch import Tensor
torch.set_grad_enabled(False)

# Generate Data in a disk of radius 1/sqrt(2 * Pi) 
input, target = generate_disc_dataset(2000)
train_input, train_target, test_input, test_target = split_dataset(input, target, 0.5)
train_input, test_input = normalize_data(train_input, test_input)


# Creates a model with 2 input units, 1 output unit and 3 hidden Layers with 25 units each 
model_sgd = Sequential([Linear(2, 25), ReLU(), Linear(25, 25), ReLU(), Linear(25, 25), ReLU(), Linear(25, 1)])

# Train it using MSE Loss and SGD optimizer
model_sgd = train_SGD(model_sgd, train_input, train_target, learning_rate=5e-5, nb_epochs=500, print_= True)

print("Performances using SGD optimizer")
print("On train :")
accuracy_one_output(model_sgd, train_input, train_target)
print("On test :")
accuracy_one_output(model_sgd, test_input, test_target)