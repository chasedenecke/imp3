
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import seaborn as sns
import math
sns.set()

import sys
import numpy as np
import matplotlib.pyplot as plt

# Function returns two subsets of a data set.
# @param split A % to split the data on.
# @param dataset A dataset.
def DataSplit(split, dataset):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    #subsetSplit is the location of the split.
    subsetSplit = int(np.floor(split * dataset_size))
    # Genereate indices for the two subsets.
    subset1_indices, subset2_indices = indices[subsetSplit:], indices[:subsetSplit]
    # Fill the subsets.
    subset2_sampler = torch.utils.data.Subset(dataset, subset2_indices)
    subset1_sampler = torch.utils.data.Subset(dataset, subset1_indices)

    return subset1_sampler, subset2_sampler


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print('Using PyTorch version:', torch.__version__, ' Device:', device)

batch_size = 32
# For gray scaling.
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor()
    ])

print("Loading data set.")
train_dataset = datasets.CIFAR10('./data',
                               train=True,
                               download=True,
                               transform=transform)

test_dataset = datasets.CIFAR10('./data',
                               train=False,
                               download=True,
                               transform=transform)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
        batch_size = batch_size,
        shuffle=True)

print("Generating validation sample")
validation_sampler, train_sampler = DataSplit(0.8, train_dataset)

validation_loader = torch.utils.data.DataLoader(dataset=validation_sampler,
        batch_size = batch_size,
        shuffle=True)

print("Generating 5 testing samples.")
training_data = []
# Frequency of the split to produce 4 even sets.
split =[0.75, 0.66667, 0.5]
print(len(train_sampler))
for i in range(0, 3):
    subset, train_sampler = DataSplit(split[i], train_sampler)
    training_data.append(subset)
training_data.append(train_sampler)

for i, elem in enumerate(training_data):
    print("traning set ", i + 1 ," size: ", len(elem))

train_loader = []
for elem in training_data:
   tmp = torch.utils.data.DataLoader(dataset=elem,
            batch_size = batch_size,
            shuffle=True)
   train_loader.append(tmp)

print("\n")
for i, loader in enumerate(train_loader):
    print("Dataset: ", i + 1)
    for (X_train, y_train) in loader:
        print('X_train:', X_train.size(), 'type:', X_train.type())
        print('y_train:', y_train.size(), 'type:', y_train.type())
        print("\n")
        break

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(32*32, 100)
        self.fc1_drop = nn.Dropout(0.2)
        self.fc3 = nn.Linear(100, 10)

    def forward(self, x):
        x = x.view(-1, 32*32)
        x = F.relu(self.fc1(x))
        x = self.fc1_drop(x)
        return F.log_softmax(self.fc3(x), dim=1)

model = Net().to(device)
lr = float(sys.argv[1])
optimizer = torch.optim.SGD(model.parameters(), lr, momentum=0.5)
criterion = nn.CrossEntropyLoss()

print(model)

def train(epoch, loader, log_interval=200):
    # Set model to training mode
    model.train()
    # Loop over each batch from the training set
    for batch_idx, (data, target) in enumerate(loader):
        # Copy data to GPU if needed
        data = data.to(device)
        target = target.to(device)

        # Zero gradient buffers
        optimizer.zero_grad()
        # Pass data through the network
        output = model(data)
        # Calculate loss
        loss = criterion(output, target)

        # Backpropagate
        loss.backward()

        # Update weights
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(loader.dataset),
                100. * batch_idx / len(loader), loss.data.item()))

def validate(loss_vector, accuracy_vector, loader):
     model.eval()
     val_loss, correct = 0, 0
     for data, target in loader:
         data = data.to(device)
         target = target.to(device)
         output = model(data)
         val_loss += criterion(output, target).data.item()
         pred = output.data.max(1)[1] # get the index of the max log-probability
         correct += pred.eq(target.data).cpu().sum()

     val_loss /= len(loader)
     loss_vector.append(val_loss)

     accuracy = 100. * correct.to(torch.float32) / len(loader.dataset)
     accuracy_vector.append(accuracy)

     print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
         val_loss, correct, len(loader.dataset), accuracy))

epochs = 5

lossv, accv = [], []
for i, dataset in enumerate(train_loader):
    print("Using dataset ", i + 1)
    for epoch in range(1, epochs + 1):
        train(epoch, dataset)
        validate(lossv, accv, validation_loader)

#print("lossv size: ", len(lossv))
#print("accv size: ", len(accv))
#print("loss: ",lossv[0])

#plt.figure(figsize=(5,3))
plt.subplot(2, 1, 1)
plt.plot(np.arange(1,(epochs*4)+1), lossv, 'b-')
plt.title('validation loss')

plt.subplot(2, 1, 2)
#plt.figure(figsize=(5,3))
plt.plot(np.arange(1,(epochs*4)+1), accv, 'r-')
plt.title('validation accuracy');

plt.subplots_adjust(hspace=0.5)

losst, acct = [], []
print("Testing data results")
validate(losst, acct, test_loader)
print("Accuracy of test")
print(acct)

try:
    plt.show()
except:
    print("Cannot show graph.")

print("saving graph in p2.png")
plt.savefig('p3.png')
