import torch
import torch.nn as nn
import torch.nn.functional as F

class MNIST_model(nn.Module):
    def __init__(self, hidden_size=64):
        super(MNIST_model, self).__init__()

        self.hidden_size = hidden_size
        self.hidden_layer = nn.Linear(28*28, hidden_size)
        self.output_layer = nn.Linear(hidden_size, 10)
        self.sm_layer = nn.Linear(hidden_size, hidden_size) #self-modeling layer

    def forward(self, x):
        a = torch.flatten(x, start_dim=1) #dim=0 is batch dimension
        a = self.hidden_layer(a)
        x = F.relu(a) #outputs of hidden layer, should this be saved before or after activation function(relu)?

        output = self.output_layer(x)
        output = F.softmax(output, dim=1)

        a_hat = self.sm_layer(a)

        return output, a_hat, a