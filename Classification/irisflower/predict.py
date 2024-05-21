from model import Model
import torch

# initialize model
model_class = Model()

# load trained weights and bias to newral network
model_class.load_state_dict(torch.load("iris_model.pt"))

# data
data = [6.7,3.3,5.7,2.1]

# predict values with no backpropagation
with torch.no_grad():
    data = torch.tensor(data)
    pred = model_class(data)
    predicted_item = pred.argmax().item()

    if predicted_item == 0:
        print("Setosa")
    elif predicted_item == 1:
        print("Virginica")
    elif predicted_item == 2:
        print("Versicolor")