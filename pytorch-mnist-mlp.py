
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import seaborn as sns
import math
sns.set()

import numpy as np
import matplotlib.pyplot as plt

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    
print('Using PyTorch version:', torch.__version__, ' Device:', device)

batch_size = 32

train_dataset = datasets.CIFAR10('./data', 
                               train=True, 
                               download=True, 
                               transform=transforms.ToTensor())

validation_split = 0.8
dataset_size = len(train_dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
print("split = ", split)
validation_indices, train_indices = indices[split:], indices[:split]
print("train indices = ", train_indices)
print("validation indices = ", validation_indices)
train_sampler = torch.utils.data.Subset(train_dataset, train_indices)
validation_sampler = torch.utils.data.Subset(train_dataset, validation_indices)
print("lenghth of train = ", len(train_sampler))


# subset_loader = torch.utils.data.Subset(train_dataset, batch_size=32, shuffle=True, sampler=SubsetRandomSampler())
# print("length of subset loader = ", len(subset_loader))

# validation_dataset = datasets.CIFAR10('./data', 
#                                     train=False, 
#                                     transform=transforms.ToTensor())

# train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
#                                            batch_size=batch_size, 
#                                            shuffle=True)

# validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, 
#                                                 batch_size=batch_size, 
#                                                 shuffle=False)


# for (X_train, y_train) in train_loader:
#     print('X_train:', X_train.size(), 'type:', X_train.type())
#     print('y_train:', y_train.size(), 'type:', y_train.type())
#     break

# pltsize=1
# plt.figure(figsize=(10*pltsize, pltsize))

# for i in range(10):
#     plt.subplot(1,10,i+1)
#     plt.axis('off')
#     plt.imshow(X_train[i,:,:,:].numpy().reshape(32,32,3), cmap="gray")
#     plt.title('Class: '+str(y_train[i].item()))

# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.fc1 = nn.Linear(32*32, 50)
#         self.fc1_drop = nn.Dropout(0.2)
#         self.fc2 = nn.Linear(50, 50)
#         self.fc2_drop = nn.Dropout(0.2)
#         self.fc3 = nn.Linear(50, 10)

#     def forward(self, x):
#         x = x.view(-1, 32*32)
#         x = F.relu(self.fc1(x))
#         x = self.fc1_drop(x)
#         x = F.relu(self.fc2(x))
#         x = self.fc2_drop(x)
#         return F.log_softmax(self.fc3(x), dim=1)

# model = Net().to(device)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
# criterion = nn.CrossEntropyLoss()

# print(model)

# def train(epoch, log_interval=200):
#     # Set model to training mode
#     model.train()
    
#     # Loop over each batch from the training set
#     for batch_idx, (data, target) in enumerate(train_loader):
#         # Copy data to GPU if needed
#         data = data.to(device)
#         target = target.to(device)

#         # Zero gradient buffers
#         optimizer.zero_grad() 
        
#         # Pass data through the network
#         output = model(data)

#         # Calculate loss
#         loss = criterion(output, target)

#         # Backpropagate
#         loss.backward()
        
#         # Update weights
#         optimizer.step()
        
#         if batch_idx % log_interval == 0:
#             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                 epoch, batch_idx * len(data), len(train_loader.dataset),
#                 100. * batch_idx / len(train_loader), loss.data.item()))

# def validate(loss_vector, accuracy_vector):
#     model.eval()
#     val_loss, correct = 0, 0
#     for data, target in validation_loader:
#         data = data.to(device)
#         target = target.to(device)
#         output = model(data)
#         val_loss += criterion(output, target).data.item()
#         pred = output.data.max(1)[1] # get the index of the max log-probability
#         correct += pred.eq(target.data).cpu().sum()

#     val_loss /= len(validation_loader)
#     loss_vector.append(val_loss)

#     accuracy = 100. * correct.to(torch.float32) / len(validation_loader.dataset)
#     accuracy_vector.append(accuracy)
    
#     print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
#         val_loss, correct, len(validation_loader.dataset), accuracy))

# epochs = 5

# lossv, accv = [], []
# for epoch in range(1, epochs + 1):
#     train(epoch)
#     validate(lossv, accv)

# plt.figure(figsize=(5,3))
# plt.plot(np.arange(1,epochs+1), lossv)
# plt.title('validation loss')

# plt.figure(figsize=(5,3))
# plt.plot(np.arange(1,epochs+1), accv)
# plt.title('validation accuracy');