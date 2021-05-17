
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