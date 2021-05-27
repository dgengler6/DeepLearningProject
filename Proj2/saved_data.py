# Activations 

r = ReLU()
th = Tanh()
s = Sigmoid()

t = torch.FloatTensor([-10, 0, 1, 3, 10])

print(r.forward(t))
print(r.backward(t))
print(th.forward(t))
print(th.backward(t))
print(s.forward(t))
print(s.backward(t))


# Linear 


l = Linear(10, 10)
t = torch.empty(10).fill_(1)
l.forward(t)
l.backward(t)

# Sequential 

l1 = Linear(1,2)
l2 = Linear(2,2)
l3 = Linear(2,1)
print(l1.param(), l2.param(), l3.param())
s = Sequential([l1, l2, l3])
s.param()

# Losses 

# Test

target = torch.FloatTensor([1, 1, 1])
pred = torch.FloatTensor([0, 1, 3])
lossMSE = LossMSE()
print(lossMSE.forward(pred, target))
print(lossMSE.backward())
# Test

target = torch.FloatTensor([1, 1, 1])
pred = torch.FloatTensor([0, 1, 3])
lossMAE = LossMAE()
print(lossMAE.forward(pred, target))
print(lossMAE.backward())
# Test

target = torch.FloatTensor([0, 0, 1, 1])
pred = torch.FloatTensor([0.3, 0.1, 0.6, 0.1])
lossBCE = LossBCE()
print(lossBCE.forward(pred, target))
print(lossBCE.backward())

# Optimizers 

#  SGD
l1 = Linear(1,2)
l2 = Linear(2,2)
l3 = Linear(2,1)
print("l1", l1.param(), "l2", l2.param(), "l3", l3.param())
s = Sequential([l1, l2, l3])

sgd = SGD(s.param(), 0.001)
sgd.show()
sgd.step()

#  Adam 

l1 = Linear(1,2)
l2 = Linear(2,2)
l3 = Linear(2,1)
print("l1", l1.param(), "l2", l2.param(), "l3", l3.param())
s = Sequential([l1, l2, l3])

adam = Adam(s.param(), 0.001)
adam.show()


l1 = Linear(1,2)
l2 = Linear(2,2)
l3 = Linear(2,1)
print("l1", l1.param(), "l2", l2.param(), "l3", l3.param())
s = Sequential([l1, l2, l3])

adam = Adam(s.param(), 0.001)
adam.step()
adam.zero_grad()

params = torch.FloatTensor([[(10, 5), (10, -20)], [(0, 10), (0, -10)]])
sgd = SGD(params , 0.1)
sgd.step()
sgd.show()
sgd.zero_grad()
sgd.show()

# Minibaatches Stuff 
def train_batches(model, train_input, train_target, learning_rate=0.001, nb_epochs=50, mini_batch_size = 500):
    
    criterion = LossMSE()
    optimizer = SGD(model.param(), learning_rate)
    
    for epoch in range(nb_epochs):
        
        avg_acc_loss = 0

        #for idx, input in enumerate(train_input):
        for b in range(0, train_input.size(0), mini_batch_size):
            print(train_input.narrow(0, b, mini_batch_size).shape)
            optimizer.zero_grad()
            output = model.forward(train_input.narrow(0, b, mini_batch_size).T)
            target = train_target.narrow(0, b, mini_batch_size)
            
            loss = criterion.forward(output, target)
            print(output, target, loss.item())
            acc_loss = acc_loss + loss.item()
            
            grad_loss = criterion.backward()
            model.backward(grad_loss)
            optimizer.step()
            
            
        
        print(f"epoch {epoch}, Loss {acc_loss}")
    return model

input, target = generate_disc_dataset(2000)
train_input, train_target, test_input, test_target = split_dataset(input, target, 0.5)
train_input, test_input = normalize_data(train_input, test_input)
plot_from_input(train_input, train_target)
# Pas sûr si on doit faire une activation fonction à la fin
model = Sequential([Linear(2, 16), ReLU(), Linear(16, 32), ReLU(), Linear(32, 32), ReLU(), Linear(32, 2)])
one_hot_targets = one_hot_encoder(train_target)

model = train_batches(model, train_input, one_hot_targets)
test_one_output(model, test_input, test_target)