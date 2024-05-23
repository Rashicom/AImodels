from model import Model
import torch

# init model
model1 = Model()

# load trained weights and bias to newral network
model1.load_state_dict(torch.load("california_model.pt"))

# data
data = [ 0.2924,  0.5055, -0.0489, -0.1421,  0.0425,  0.1512, -0.9302,  0.8222]

# predict
with torch.no_grad():
    data = torch.tensor(data)
    pred = model1.forward(data)
    print("predicted value: ", pred.item())