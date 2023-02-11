import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from torchvision.transforms import ToTensor

import numpy as np

#Randomised model to extract features

class RandomFeatures(nn.Module):
    FEATURES = 10000
    def __init__(self): 
        super(RandomFeatures, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 1024, bias=False),
            nn.ReLU(),
            nn.Linear(1024, 1024, bias=False),
            nn.ReLU(),
            nn.Linear(1024, RandomFeatures.FEATURES, bias=False),
        )
        self.linear_relu_stack.apply(self.init_weights)

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.torch.nn.init.normal_(m.weight, mean=0, std=0.4)

random_model = RandomFeatures()

def transform_to_random(element):
    element = ToTensor()(element)
    return random_model(element)

#Data

training_data = Subset(datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=transform_to_random
), np.arange(6000))

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=transform_to_random
)

train_dataloader = DataLoader(training_data, batch_size=64)

test_dataloader = DataLoader(test_data, batch_size=64)

#now do linear regression on the random features

model = nn.Linear(RandomFeatures.FEATURES, 10, bias=False)
learning_rate = 0.00002

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(torch.flatten(X, start_dim=1, end_dim=-1))
        #print("prediction:", pred.size())
        #print("y:", y)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(torch.flatten(X, start_dim=1, end_dim=-1))
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")
