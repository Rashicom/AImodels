import torch.nn as nn
import torch.nn.functional as F


# model class
class Model(nn.Module):
    def __init__(self):
        super().__init__()

        # convolutional layers
        self.conv1 = nn.Conv2d(1,6,3,1)
        self.conv2 = nn.Conv2d(6,16,3,1)

        # fully connected layers
        # input = 16*5*5 (16-filters images, each image dimention- 5 * 5)
        self.fc1 = nn.Linear(16*5*5,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10) # 10 outputs
    
    def forward(self,x):
        x = F.relu(self.conv1(x)) # passing image through first layer and rectifying
        x = F.max_pool2d(x,2,2,) # pooling image using 2 * 2 kernal to reduce memory

        x = F.relu(self.conv2(x)) # passing image through second layer and rectifying
        x = F.max_pool2d(x,2,2,) # pooling image using 2 * 2 kernal to reduce memory

        # converting n(number of images)*16*5*5 shaped tensor into 1-d matrices to pass to fully connected later
        # -1 is all images we pass, multiple images can ba passed to input later
        x = x.view(-1,16*5*5)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x) # output without rectifications
        return F.log_softmax(x,dim=1)










