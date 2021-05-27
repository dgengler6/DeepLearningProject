import torch
import math
import time
import datetime
import random

from torch import Tensor
torch.set_grad_enabled(False)

# The father of all Modules
class Module(object):
    def forward(self, *input):
        raise NotImplementedError
        
    def backward(self, *gradwrtoutput):
        raise NotImplementedError
        
    def param(self):
        return []
    

# Activation functions 

# Implements a ReLu layer
class ReLU(Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, input):
        self.t = input
        return input.clamp(0)

    def backward(self, gradwrtoutput):
        input = self.t
        sign = input.sign().clamp(0)
        return sign * gradwrtoutput
        
    def param(self):
        return []
    
# Implements a Tanh Layer 
class Tanh(Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, input):
        self.t = input
        
        out = []
        for x in input :
            e = math.exp(-2 * x)
            val = (1 - e) / (1 + e)
            out.append(val)
        return torch.FloatTensor(out)

    def backward(self, gradwrtoutput):
        z = self.t
        e = torch.exp(-2 * z)
        d_tanh = 4 * e / (1 + e)**2
        return d_tanh * gradwrtoutput
        
    def param(self):
        return []
    
    
# Implements a Sigmoid Layer 
class Sigmoid(Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, input):
        self.t = input
        return input.mul(-1).exp().add(1).pow(-1)

    def backward(self, gradwrtoutput):
        sig = self.t.mul(-1).exp().add(1).pow(-1)
        return sig.mul(-1).add(1).mul(sig)
        
    def param(self):
        return []
    
    
# Layers 

# Linear Layer 
class Linear(Module):
    def __init__(self, input_size, output_size, mean=0, std=1):
        super().__init__()
        self.W = torch.empty(output_size, input_size).normal_(mean, std)
        self.b = torch.empty(output_size).normal_(mean, std)
        self.dW = torch.zeros(output_size, input_size)
        self.db = torch.zeros(output_size)
    def forward(self, input):
        self.t = input
        return self.W.mv(input).add(self.b)

    def backward(self, gradwrtoutput):
        self.dW.add_(gradwrtoutput.view(1, -1).t().mm(self.t.view(1, -1)))
        self.db.add_(gradwrtoutput)
        #print(gradwrtoutput.shape, self.W.shape)
        return self.W.t().mv(gradwrtoutput)
        
        
    def param(self):
        return [(self.W, self.dW), (self.b, self.db)]
    
    
# Sequential 

class Sequential(Module):
    def __init__(self, module_list):
        super().__init__()
        self.modules = module_list
        
    def forward(self, input):
        x = input
        for module in self.modules :
            x = module.forward(x)
        return x 

    def backward(self, gradwrtoutput):
        x = gradwrtoutput
        for module in reversed(self.modules) :
            x = module.backward(x)
        return x
        
    def param(self):
        params = []
        for module in self.modules:
            for par in module.param():
                params.append(par)
        return params
    
# Losses 

# MSE 
class LossMSE(Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, pred, target):
        self.pred = pred
        self.target = target
        return (pred - target.float()).pow(2).mean()
    
    # J'ai pas mis les arguments dans backward car je pars du principe que la MSE est la dernière fonction utilisée
    def backward(self):
        return 2 * (self.pred - self.target)
    
    def param(self):
        return []

# MAE 
class LossMAE(Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, pred, target):
        self.pred = pred
        self.target = target
        return (pred - target.float()).abs().mean()

    def backward(self):
        error = self.pred - self.target
        return error.sign()
        
    def param(self):
        return []
    
    
# Binary cross entropy loss
# Note: We need to use sigmoid function before using BCE otherwise it makes no sense
class LossBCE(Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, pred, target):
        self.pred = pred
        self.target = target
        return ((target * pred.log()) + ((1 - target) * (1 - pred).log())).mean()

    def backward(self):
        return ((-target/pred) + ((1 - target)/(1 - pred))).mean()
        
    def param(self):
        return []
    
# Optimizers 

# SGD
class SGD():
    def __init__(self, params, lr, seq = True):
        self.params = params
        self.lr = lr
        self.seq = seq
    def step(self): 
        for module in self.params:
            param, grad = module
            if (param is not None) and (grad is not None):
                param.sub_(grad, alpha=self.lr)
    
    def zero_grad(self):
        for module in self.params:
            param, grad = module
            if (param is not None) and (grad is not None):
                grad.zero_()
                    
    def show(self):
        # To remove, for debugging
        for idx, i in enumerate(self.params):
            print(idx, i)

# Adam optimizer 
class Adam():
    def __init__(self, params, lr, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-8):
        self.params = params
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        
        # Adam inner values 
        self.t = 0
        self.adam_values = []
        for t in self.params:
            t0, t1 = t
            self.adam_values.append((torch.zeros_like(t0), torch.zeros_like(t1)))

    def step(self):
        self.t += 1 
        for idx_module, module in enumerate(self.params):
            param, grad = module
            if (param is not None) and (grad is not None):
                #Update Vd and Sd, for each tuple we keep vd and sd 
                vd, sd = self.adam_values[idx_module]

                vd.mul_(self.beta_1).add_(grad.mul(1 - self.beta_1))
                sd.mul_(self.beta_2).add_(grad.pow(2).mul(1 - self.beta_2))

                # Compute corrected Values
                vd_corr = vd.mul(torch.tensor([[self.beta_1]]).pow(self.t).mul(-1).add(1).pow(-1)) 
                sd_corr = sd.mul(torch.tensor([[self.beta_2]]).pow(self.t).mul(-1).add(1).pow(-1)) 

                # Update the parameter 
                v =sd_corr.sqrt().add(self.epsilon).pow(-1).mul(vd_corr)
                
                param.sub_(v.reshape(param.shape), alpha=self.lr)

    def zero_grad(self):
        for module in self.params:
            param, grad = module
            if (param is not None) and (grad is not None):
                grad.zero_()
                    
    def show(self):
        # To remove, for debugging
        print(self.t)
        print("PARAMS : ")
        for i in self.params:
            print(i)
        print(("ADAM [Vd, Sd] : "))
        for i in self.adam_values:
            print(i)