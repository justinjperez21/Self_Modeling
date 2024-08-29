import torch
import torchvision
import torch.nn.functional as F
from tqdm.auto import tqdm
from MNIST_model import *
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE_TRAIN = 64 #hyperparameters from paper
BATCH_SIZE_TEST = BATCH_SIZE_TRAIN

train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('./datasets', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=BATCH_SIZE_TRAIN, shuffle=True)

test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('./datasets', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=BATCH_SIZE_TRAIN, shuffle=True)

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
    loss_c_list = []
    loss_s_list = []
    acc_list = []
    for inputs, targets in pbar:
        optimizer.zero_grad(set_to_none=True)
        inputs, targets = inputs.to(device), targets.to(device)

        outputs, a_hat, a = model(inputs) #a_hat and a are as defined in the paper

        curr_loss_c = loss_c(outputs, targets)
        curr_loss_s = loss_s(a_hat, a)/model.hidden_size
        curr_loss_c /= BATCH_SIZE_TRAIN
        curr_loss_s /= BATCH_SIZE_TRAIN

        loss = curr_loss_c + w_s*curr_loss_s
        #loss /= BATCH_SIZE_TRAIN #convert to mean

        loss_c_list.append(curr_loss_c.item())
        loss_s_list.append(curr_loss_s.item())

        loss.backward()
        optimizer.step()

        acc = get_total_correct(outputs, targets)/BATCH_SIZE_TRAIN
        acc_list.append(acc)

        pbar.set_description(f"Training | Loss_c: {curr_loss_c}, Loss_s: {curr_loss_s}, Loss: {loss}, Accuracy: {acc}")
    
    return loss_c_list, loss_s_list, acc_list

def test(model, device, test_loader, w_s=1):
    model.eval().to(device)
    #pbar = tqdm(test_loader)
    tot_loss_c = 0
    tot_loss_s = 0
    correct = 0
    with torch.no_grad(): #no_grad to save on computation
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs, a_hat, a = model(inputs)

            tot_loss_c += loss_c(outputs, targets).item() #add up to calculate mean later
            tot_loss_s += loss_s(a_hat, a).item()/model.hidden_size
            #tot_loss += curr_loss_c + w_s*curr_loss_s
            
            correct += get_total_correct(outputs, targets)

    loss_c_mean = tot_loss_c/len(test_loader.dataset) #convert sum to mean
    loss_s_mean = tot_loss_s/len(test_loader.dataset)
    acc = correct/len(test_loader.dataset)

    return loss_c_mean, loss_s_mean, acc

model = MNIST_model(hidden_size=64)
optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, nesterov=True) #hyperparameters from paper
optimizer.zero_grad(set_to_none=True)

w_s = 1

column_list = ['epoch', 'loss_c', 'loss_s', 'loss', 'accuracy']
train_metrics_df = pd.DataFrame(columns=column_list)
test_metrics_df = pd.DataFrame(columns=column_list)

for epoch in range(50):
    curr_train_metrics_df = pd.DataFrame(columns=column_list)
    curr_test_metrics_df = pd.DataFrame(columns=column_list)

    print(f"Starting epoch {epoch}")
    print('\n')

    train_loss_c_list, train_loss_s_list, train_acc_list = train(model, device, train_loader, optimizer, w_s=w_s)

    curr_train_metrics_df['epoch'] = len(train_loss_c_list)*[epoch]
    curr_train_metrics_df['loss_c'] = train_loss_c_list
    curr_train_metrics_df['loss_s'] = train_loss_s_list
    curr_train_metrics_df['loss'] = curr_train_metrics_df['loss_c'] + w_s*curr_train_metrics_df['loss_s']
    curr_train_metrics_df['accuracy'] = train_acc_list

    train_metrics_df = pd.concat([train_metrics_df, curr_train_metrics_df])

    print('\n')

    test_loss_c, test_loss_s, test_acc = test(model, device, test_loader, w_s=w_s)

    curr_test_metrics_df['epoch'] = [epoch]
    curr_test_metrics_df['loss_c'] = [test_loss_c]
    curr_test_metrics_df['loss_s'] = [test_loss_s]
    curr_test_metrics_df['accuracy'] = [test_acc]

    test_metrics_df = pd.concat([test_metrics_df, curr_test_metrics_df])

    print(f"Test Loss_c: {test_loss_c}, Test Loss_s: {test_loss_s}, Test Loss: {test_loss_c + w_s*test_loss_s}, Test Accuracy: {test_acc}")


train_metrics_df.to_csv(f'Train_ws{w_s}_hiddensize{model.hidden_size}.csv')
test_metrics_df['loss'] = test_metrics_df['loss_c'] + w_s*test_metrics_df['loss_s']
test_metrics_df.to_csv(f'Test_ws{w_s}_hiddensize{model.hidden_size}.csv')