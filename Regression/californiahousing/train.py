from sklearn.datasets import fetch_california_housing
import copy
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from model import Model
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Fetching california housing dataset
data = fetch_california_housing()

X, y = data.data, data.target

# train-test split of dataset
# set ranod state for reproduce the same splits insted of shuffling , good for testing
torch.manual_seed(32)
X_train_row, X_test_row, y_train, y_test = train_test_split(X,y, train_size=0.8, random_state=32)

# some rows in the data is verry far from teh mean so we standerdize it to improve ml convergence
# this is not compalsory , but it should increase the convergence
scalar = StandardScaler()
scalar.fit(X_train_row)
X_train = scalar.transform(X_train_row)
X_test = scalar.transform(X_test_row)


# split is shuffled each run
# X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.8, shuffle=True)

# converting into tensor
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1,1) # convert to 2d matrix
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1,1) # convert to 2d matrix


# trainning params
n_epochs = 100 #number of epochsto run
bacht_size = 10 #large data seperated to baches to train to reduce memory overload
batch_start = torch.arange(0, len(X_train), bacht_size) #batch list

history = []

# model instance
model1 = Model()

# loss function and optimizer
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model1.parameters(), lr=0.0001)

# plot config
plt.axis([0, n_epochs, 0, 2])

for epoch in tqdm(range(n_epochs),colour="Green"):
    model1.train()
    for start in tqdm(batch_start, desc=f"Batch Train >> ", ascii=True):
        # get batch
        X_batch = X_train[start:start+bacht_size]
        y_batch = y_train[start:start+bacht_size]

        # forward pass
        y_pred = model1.forward(X_batch)
        
        # calculate mse loss
        loss = loss_fn(y_pred,y_batch)

        # propage loss backward for update wight
        optimizer.zero_grad()
        loss.backward()
        
        # update weights
        optimizer.step()
    
    model1.eval()
    # calculte accuracy in each epoch
    y_pred = model1.forward(X_test)
    mse = loss_fn(y_pred, y_test)
    loss_val = mse.detach().numpy()
    history.append(loss_val)
    plt.scatter(epoch,loss_val)
    plt.pause(0.001)


plt.show()
print("Lowest loss: ", min(history))

# Test model
print("Tsting Started")
actual_val = []
predictions = []
model1.eval()
with torch.no_grad():
    for i, data in tqdm(enumerate(X_test),colour="Blue"):
        y_pred = model1.forward(data)
        actual_val.append(y_test[i].item())
        predictions.append(y_pred.item())

plt.plot([i for i in range(len(X_test))],actual_val, label="actual_data")
plt.plot([i for i in range(len(X_test))],predictions, label="predictions")
plt.title("Actual vs Predictions")
plt.legend()
plt.show()

# sage model
torch.save(model1.state_dict(), "california_model.pt")
