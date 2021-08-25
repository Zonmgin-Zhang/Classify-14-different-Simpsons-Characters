#!/usr/bin/env python3
"""
student.py

UNSW COMP9444 Neural Networks and Deep Learning

You may modify this file however you wish, including creating additional
variables, functions, classes, etc., so long as your code runs with the
hw2main.py file unmodified, and you are only using the approved packages.

You have been given some default values for the variables train_val_split,
batch_size as well as the transform function.
You are encouraged to modify these to improve the performance of your model.

The variable device may be used to refer to the CPU/GPU being used by PyTorch.
You may change this variable in the config.py file.
"""
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms

"""
   Answer to Question:

I designed a neural network for Simpsons Character Classification.
In each epoch of training, we first selected some images and used
neural network to predict their classifications,
and then adjusted the parameters of the neural network according to
 the loss between them and the real classifications.
Train the network repeatedly until it achieves the desired effect.

The designed neural network is composed of five convolutional neural network layers and two fully connected layers.
Convolutional neural network layers are used to extract image features,
and fully connected layers are used to map image features to 14 categories.
Because we are committed to predicting the correct category,
we choose cross entropy as the loss function and use adam as the optimizer.

In terms of image transformation,
because a given image represents a character,
and the orientation of the character is divided into left and right,
we first make the image have a probability of 0.5 to flip horizontally,
and consider that the angle of the character in the image may be different,
so we rotate the image within a certain angle interval.

In order for our training effect to be better and try to avoid over-fitting,
we adjust the parameters through experiments to get epochs, batch_size, etc.
The number of layers of the neural network, the number of neurons in each layer, train_val_split, etc. are selected based on experience.

In order to avoid overfitting,
I added a pooling layer after each convolutional neural network layer,
added a dropout layer after each fully connected layer,
and minimized the number of neural network layers and training iterations.
At the same time, preprocessing of the image (such as image rotation, horizontal flip)
can also improve the generalization ability of the model.


"""


############################################################################
######     Specify transform(s) to be applied to the input images     ######
############################################################################
def transform(mode):
    """
    Called when loading the data. Visit this URL for more information:
    https://pytorch.org/vision/stable/transforms.html
    You may specify different transforms for training and testing
    """
    if mode == 'train':
        transfer = transforms.Compose(
            [transforms.RandomHorizontalFlip(), transforms.RandomRotation(degrees=15), transforms.ToTensor()]
        )
        return transfer
    elif mode == 'test':
        return transforms.ToTensor()


############################################################################
######   Define the Module to process the images and produce labels   ######
############################################################################
class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64,256,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )

        self.linear_layer1 = nn.Linear(4096, 512)
        self.linear_layer2 = nn.Linear(512, 14)

    def forward(self, t):
        #print(t.shape)
        x = self.cnn_layers(t)
        #print(x.shape)
        x = x.view(-1, 4096)
        x = F.dropout(F.relu(self.linear_layer1(x)))
        x = self.linear_layer2(x)
        return x


class loss(nn.Module):
    """
    Class for creating a custom loss function, if desired.
    If you instead specify a standard loss function,
    you can remove or comment out this class.
    """
    def __init__(self):
        super(loss, self).__init__()

    def forward(self, output, target):
        return F.cross_entropy(output, target)


net = Network()
lossFunc = loss()
############################################################################
#######              Metaparameters and training options              ######
############################################################################
dataset = "./data"
train_val_split = 0.8
batch_size = 32
epochs = 100
optimiser = optim.Adam(net.parameters(), lr=0.001)
