import torch
import torchvision
import torch.nn.functional as F
from tqdm.auto import tqdm
from MNIST_model import *
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size_train = 64 #from paper
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
    loss_list = []
    acc_list = []
    for inputs, targets in pbar:
        optimizer.zero_grad(set_to_none=True)
        inputs, targets = inputs.to(device), targets.to(device)
        #targets = F.one_hot(targets, num_classes=10)
        outputs, a_hat, a = model(inputs) #a_hat and a are as defined in the paper
        loss = loss_c(outputs, targets) + w_s*loss_s(a_hat, a)/model.hidden_size
        loss /= batch_size_train #convert to mean
        loss_list.append(loss.item())
        loss.backward()
        optimizer.step()
        acc = get_total_correct(outputs, targets)/batch_size_train
        acc_list.append(acc)
        pbar.set_description(f"Train Loss: {loss}, Train Accuracy: {acc}")
    
    return loss_list, acc_list

def test(model, device, test_loader, w_s=1):
    model.eval().to(device)
    #pbar = tqdm(test_loader)
    loss = 0
    correct = 0
    with torch.no_grad(): #no_grad to save on computation
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, a_hat, a = model(inputs)
            loss += loss_c(outputs, targets) + w_s*loss_s(a_hat, a) #add up to calculate mean later
            correct += get_total_correct(outputs, targets)

    loss /= len(test_loader.dataset) #convert sum to mean
    correct /= len(test_loader.dataset)

    return loss.item(), correct

model = MNIST_model(hidden_size=64)
optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, nesterov=True)
optimizer.zero_grad(set_to_none=True)

w_s = 1

column_list = ['epoch', 'loss', 'accuracy']
train_metrics_df = pd.DataFrame(columns=column_list)
test_metrics_df = pd.DataFrame(columns=column_list)

for epoch in range(50):
    curr_train_metrics_df = pd.DataFrame(columns=column_list)
    curr_test_metrics_df = pd.DataFrame(columns=column_list)

    print(f"Starting epoch {epoch}")
    print('\n')

    train_loss_list, train_acc_list = train(model, device, train_loader, optimizer, w_s=w_s)
    curr_train_metrics_df['epoch'] = len(train_loss_list)*[epoch]
    curr_train_metrics_df['loss'] = train_loss_list
    curr_train_metrics_df['accuracy'] = train_acc_list
    train_metrics_df = pd.concat([train_metrics_df, curr_train_metrics_df])

    print('\n')

    test_loss, test_acc = test(model, device, test_loader, w_s=w_s)
    curr_test_metrics_df['epoch'] = [epoch]
    curr_test_metrics_df['loss'] = [test_loss]
    curr_test_metrics_df['accuracy'] = [test_acc]
    test_metrics_df = pd.concat([test_metrics_df, curr_test_metrics_df])
    print(f"Test Loss: {test_loss}, Test Accuracy: {test_acc}")

train_metrics_df.to_csv(f'Train_ws{w_s}.csv')
test_metrics_df.to_csv(f'Test_ws{w_s}.csv')