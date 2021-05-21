
# Basic CNN network 
class Net(nn.Module):
    def __init__(self, nb_hidden):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(256, nb_hidden)
        self.fc2 = nn.Linear(nb_hidden, 10)

    def forward(self, x):
        prt = False
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2))
        if prt:
            print(x.shape)
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2))
        if prt:
            print(x.shape)
        x = F.relu(self.fc1(x.view(-1, 256)))
        if prt:
            print(x.shape)
        x = self.fc2(x)
        if prt:
            print(x.shape)
        return x

def train_model(model, train_input, train_target, mini_batch_size, nb_epochs = 100):
    criterion = nn.CrossEntropyLoss()
    eta = 1e-3
    for e in range(nb_epochs):
        acc_loss = 0

        for b in range(0, train_input.size(0), mini_batch_size):
            output = model(train_input.narrow(0, b, mini_batch_size))
            loss = criterion(output, train_target.narrow(0, b, mini_batch_size))
            acc_loss = acc_loss + loss.item()
 
            model.zero_grad()
            loss.backward()

            with torch.no_grad():
                for p in model.parameters():
                    p -= eta * p.grad

        print(e, acc_loss)
        
def compute_nb_errors(model, input, target, mini_batch_size):
    nb_errors = 0

    for b in range(0, input.size(0), mini_batch_size):
        output = model(input.narrow(0, b, mini_batch_size))
        _, predicted_classes = output.max(1)
        for k in range(mini_batch_size):
            if target[b + k, predicted_classes[k]] <= 0:
                nb_errors = nb_errors + 1

    return nb_errors

model = Net(200)

train_input_first = train_input[:,0,:,:].reshape([1000,1,14,14])
train_model(model, train_input_first, train_classes[:,0], mini_batch_size=200, nb_epochs=100)

test_input_first = test_input[:,0,:,:].reshape([1000,1,14,14])
test_classes_first = prologue.convert_to_one_hot_labels(test_input_first, test_classes[:,0])
nb_test_errors = compute_nb_errors(model, test_input_first, test_classes_first, mini_batch_size=200)
print('test error Net {:0.2f}% {:d}/{:d}'.format((100 * nb_test_errors) / test_input_first.size(0),
                                                      nb_test_errors, test_input_first.size(0)))
# End of Model 1 

# CNN with 1 output for each digit 
class Comparisson_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(256, 200)
        self.fc2 = nn.Linear(200, 10)
    
    def cnn(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2))
        x = F.relu(self.fc1(x.view(-1, 256)))
        x = self.fc2(x)
        return x
    
    def forward(self, x):
        s = x.shape
        #print(s)
        input_1 = x[:,0,:,:].reshape([s[0],1,s[2],s[3]])
        input_2 = x[:,1,:,:].reshape([s[0],1,s[2],s[3]])
        #print(input_1.shape, input_2.shape)
        output_1 = self.cnn(input_1)
        output_2 = self.cnn(input_2)
        
        return output_1, output_2
    
def train_model2(model, train_input, train_target, mini_batch_size, nb_epochs = 100):
    criterion = nn.CrossEntropyLoss()
    eta = 1e-3
    for e in range(nb_epochs):
        acc_loss = 0

        for b in range(0, train_input.size(0), mini_batch_size):
            o1, o2 = model(train_input.narrow(0, b, mini_batch_size))
            t1, t2 = train_target.narrow(0, b, mini_batch_size)[:,0], train_target.narrow(0, b, mini_batch_size)[:,1]
            loss1 = criterion(o1, t1)
            loss2 = criterion(o2, t2)
            acc_loss = acc_loss + loss1.item() + loss2.item()
 
            model.zero_grad()
            loss1.backward()
            loss2.backward()

            with torch.no_grad():
                for p in model.parameters():
                    p -= eta * p.grad

        print(e, acc_loss)
        
def compute_nb_errors2(model, input, target, mini_batch_size):
    nb_errors = 0

    for b in range(0, input.size(0), mini_batch_size):
        o1, o2 = model(input.narrow(0, b, mini_batch_size))
        _, pc1 = o1.max(1)
        _, pc2 = o2.max(1)
        for k in range(mini_batch_size):
            if target[b + k].tolist() != torch.stack((pc1[k],pc2[k])).tolist():
                nb_errors = nb_errors + 1

    return nb_errors

model2 = Comparisson_Net()


train_model2(model2, train_input, train_classes, mini_batch_size=200, nb_epochs=100)


nb_test_errors = compute_nb_errors2(model2, test_input, test_classes, mini_batch_size=200)
print('test error Net {:0.2f}% {:d}/{:d}'.format((100 * nb_test_errors) / test_input_first.size(0),
                                                      nb_test_errors, test_input_first.size(0)))

# Train functions using MSE (worse performance than CrossEntropyLoss 

def train_model_simple_net(model, train_input, train_target, mini_batch_size, nb_epochs = 100, use_optimizer= None, _print=False):
    criterion = nn.MSELoss()
    eta = 1e-3
    if use_optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=eta)
    if use_optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=eta)
    for e in range(nb_epochs):
        acc_loss = 0

        for b in range(0, train_input.size(0), mini_batch_size):
            output = model(train_input.narrow(0, b, mini_batch_size))
            target = train_target.narrow(0, b, mini_batch_size).reshape(output.shape).float()
            
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

def train_model_auxiliary_loss(model, train_input, train_target, train_classes, mini_batch_size, nb_epochs = 100, use_optimizer= None, _print=False):
    criterion_auxilary = nn.CrossEntropyLoss()
    criterion_final = nn.MSELoss()
    
    eta = 1e-3
    if use_optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=eta)
    if use_optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=eta)
    for e in range(nb_epochs):
        acc_loss = 0

        for b in range(0, train_input.size(0), mini_batch_size):
            digit_1, digit_2, comparison = model(train_input.narrow(0, b, mini_batch_size))
            
            target_comparison = train_target.narrow(0, b, mini_batch_size).reshape(comparison.shape).float()
            
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
# First Benchmark 

# Benchmark of the basic network with Adam optimizer
nb_trials = 10
N = 1000
performances = []
for trial in range(nb_trials):
    
    # Generate Data 
    train_input, train_target, train_classes, test_input, test_target, test_classes = prologue.generate_pair_sets(N)
    train_target_one_hot = prologue.convert_to_one_hot_labels(train_input, train_target)
    test_target_total = prologue.convert_to_one_hot_labels(test_input, test_target)
    
    # Define the model 
    model_total = Simple_Net()
    
    # Train the model
    train_model_simple_net(model_total, train_input, train_target_one_hot, mini_batch_size=250, 
                      nb_epochs=25, use_optimizer="adam")
    
    # Evaluate performances 
    nb_test_errors = compute_nb_errors_simple_net(model_total, test_input, test_target_total, mini_batch_size=250)
    print('test error Net {:d} {:0.2f}% {:d}/{:d}'.format(trial, (100 * nb_test_errors) / test_input.size(0),
                                                          nb_test_errors, test_input.size(0)))
    performances.append(nb_test_errors)
    
mean_perf = 100 * sum(performances) / (N * nb_trials)
print(f"Average precision of this architecture {mean_perf}")
# Random Utility functions 

# This one displays digits 
import matplotlib.pyplot as plt

print(train_input.shape)

fig ,axes = plt.subplots(10, 2)
for i in range(10):
    first = train_input[i][0,:,:]
    second = train_input[i][1,:,:]
    first_label = train_classes[i][0]
    second_label = train_classes[i][1]
    axes[i,0].imshow(first, cmap='gray')
    axes[i,0].set_ylabel(str(first_label.item()))
    axes[i,1].imshow(second, cmap='gray', interpolation='none')
    axes[i,1].set_ylabel(str(second_label.item()))
    axes[i,0].set_xticks([])
    axes[i,0].set_yticks([])
    axes[i,1].set_xticks([])
    axes[i,1].set_yticks([])
    fig

# this one returns comparison for some digits 
for i in range(10):
    input_to_test = test_input[i]
    first_label = test_classes[i][0]
    second_label = test_classes[i][1] 
    s = input_to_test.shape
    output = model_total(input_to_test.reshape([1,s[0], s[1], s[2]]))
    _, predicted_classes = output.max(1)
    print(f"Predicted : {first_label} {'>' if predicted_classes.item() == 0 else '<'} {second_label}")