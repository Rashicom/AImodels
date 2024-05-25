import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from PIL import Image
from model import Model

# seed
torch.manual_seed(36)
model1 = Model()

# load data, 60k for training and 10k for testing
transform = transforms.ToTensor()
training_data = datasets.MNIST(root="Classification/mnist_cnn",train=True, download=True, transform=transform) #(x_train, y_train)
testing_data = datasets.MNIST(root="Classification/mnist_cnn",train=False, download=True, transform=transform) #(x_test, y_test)

# batch data for training 6k batch of 10 data for train and 1k batch of 10 data for test
train_loader = DataLoader(training_data, batch_size=10, shuffle=True) #(x_train, y_train)
test_loader = DataLoader(testing_data, batch_size=10, shuffle=False) #(x_test, y_test)


# epochs
epochs = 5

# loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model1.parameters(), lr=0.001)
loss_history = []

# plot config
# plt.axis([0, 6000, 0, 4])
iter = 0

for epoch in tqdm(range(epochs)):
    model1.train()
    # train batches
    # control number of training data using i.
    for i,(X_train,y_train) in tqdm(enumerate(train_loader),desc="Batch Train", ascii=True, colour="Green"):
        
        # in each pass we pass 10 data in X_train(in 1 batch 10 images)
        y_pred = model1.forward(X_train) # pass data to cnn
        loss = loss_fn(y_pred,y_train) # calculate loss
        loss_history.append(loss.item())
        
        # update parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # plot
        iter += 1
        plt.scatter(iter,loss.item())
        plt.pause(0.001)


# save model
torch.save(model1.state_dict(),"mnist_model.pt")




