import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from model import Model


# create model instance
model1 = Model()
torch.manual_seed(32)

# load data
url = "https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv"
data = pd.read_csv(url)
data["variety"] = data["variety"].replace("Setosa",0.0)
data["variety"] = data["variety"].replace("Virginica",1.0)
data["variety"] = data["variety"].replace("Versicolor",2.0)

# split data for testing and training
# For training
X = data.drop("variety", axis=1)

# For testing
y = data["variety"]

# converting into numpy arrays
X = X.values
y = y.values

# splid data fro testing and trainging 20 percent for test 80 percent for train
# using scikitlern
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=32)

# convert into float tensors
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)


criterion = nn.CrossEntropyLoss() #loss function
optimizer = torch.optim.Adam(model1.parameters(), lr=0.01)


# Train model
# set epoch(one run through all data)
epochs = 100
losses = []
for i in range(epochs):
    # go forward and get prediction
    y_pred = model1.forward(X_train)

    # track the losses
    loss = criterion(y_pred, y_train)
    losses.append(loss.detach().numpy())
    
    # print all data
    if i % 10 == 0:
        print(f"Epoch: {i}, loss: {loss}")

    # back propagation: take error rate and feed back to model fo fine tune weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# testing model without backpropagation
predictions = []
actual_results = []
with torch.no_grad():
    for i, data in enumerate(X_test):
        y_eval = model1.forward(data)
        
        loss = criterion(y_eval,y_test[i])
        predicted_item = y_eval.argmax().item()
        actual = y_test[i].item()
        # print(f"predicted_item:{predicted_item}, actual:{actual} loss:{loss}")
        predictions.append(predicted_item)
        actual_results.append(actual)

# save model
torch.save(model1.state_dict(),"iris_model.pt")