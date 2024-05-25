from model import Model
import torch

# initialize model
model_class = Model()

# load trained weights and bias to newral network
model_class.load_state_dict(torch.load("mnist_model.pt"))

# data
data = []

# predict values with no backpropagation
with torch.no_grad():
    data = torch.tensor(data)
    pred = model_class(data)
    predicted_item = pred.argmax().item()