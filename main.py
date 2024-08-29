import torch
import torchvision
import torch.nn.functional as F
from tqdm.auto import tqdm
from MNIST_model import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size_train = 64 #hyperparameter from paper
batch_size_test = batch_size_train

train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('./datasets', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('./datasets', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_test, shuffle=True)

def loss_c(outputs, targets): #loss for main task
    return F.cross_entropy(outputs, targets, reduction='sum') #use sum reduction for use in test function later

def loss_s(a_hat, a): #loss for self modeling task
    return F.mse_loss(a, a_hat, reduction='sum') #self modeling loss proposed in paper is just MSE

def get_total_correct(outputs, targets):
    pred = outputs.argmax(dim=1, keepdim=True) #max softmax
    correct = pred.eq(targets.view_as(pred)).sum().item()
    return correct

def train(model, device, train_loader, optimizer, w_s=1):
    model.train().to(device)
    pbar = tqdm(train_loader)
    for inputs, targets in pbar:
        optimizer.zero_grad(set_to_none=True)
        inputs, targets = inputs.to(device), targets.to(device)
        #targets = F.one_hot(targets, num_classes=10)
        outputs, a_hat, a = model(inputs) #a_hat and a are as defined in the paper
        loss = loss_c(outputs, targets) + w_s*loss_s(a_hat, a)
        loss /= batch_size_train #convert to mean
        loss.backward()
        optimizer.step()
        correct = get_total_correct(outputs, targets)
        pbar.set_description(f"Train Loss: {loss}, Train Accuracy: {correct/batch_size_train}")

def test(model, device, test_loader, w_s=1):
    model.eval().to(device)
    #pbar = tqdm(test_loader)
    loss = 0
    correct = 0
    with torch.no_grad(): #no_grad to save on computation
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, a_hat, a = model(inputs)
            loss += loss_c(outputs, targets) + w_s*loss_s(a_hat, a) #add up all losses to calculate mean loss
            correct += get_total_correct(outputs, targets)

    loss /= len(test_loader.dataset) #convert sum loss to mean
    correct /= len(test_loader.dataset)

    return loss, correct

print('test_again')

model = MNIST_model(hidden_size=64)
optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, nesterov=True) #hyperparameters from paper
optimizer.zero_grad(set_to_none=True)

w_s = 1

for epoch in range(50):
    print(f"Starting epoch {epoch}")
    print('\n')

    train(model, device, train_loader, optimizer, w_s=w_s)

    print('\n')

    test_loss, test_acc = test(model, device, test_loader, w_s=w_s)
    print(f"Test Loss: {test_loss}, Test Accuracy: {test_acc}")