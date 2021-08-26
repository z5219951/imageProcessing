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
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

"""
   Answer to Question:

Briefly describe how your program works, and explain any design and training
decisions you made along the way.
#Team Id: g024874
Team information:
    Yue Qi z5219951
    Mohammed Musa z5284114

Our model is a convolutional neural network model based on LeNet-5. In the beginning,
we set up the neural network as follows. Layer 1 is a convolution with 3 input channels, 12 output channels, kernel size in 5 and ReLu activation function. 
Layer 2 is Max Pooling with kernel size 2.  Layer 3 is a convolution with 12 input channels, 42 output channels, kernel size in 5 and ReLu activation function. 
Layer 4 is Max Pooling with kernel size 2. Layer 5 is a convolution with 42 input channels, 250 output channels, kernel size in 5 and ReLu activation function. 
Layer 6 is Max Pooling with kernel size 2. Layer 7 is Linear with 4000 input size, 600 output size and ReLu activation function.  
Layer 8 is Linear with 600 input size, 84 output size and ReLu activation function. 
Layer 9 is Linear with 84 input size, 14 output size and no activation function. 
The loss function is cross-entropy. The optimiser is Adam. 
The value of the train and validation set rate is 0.8. The batch size is 100.

There were several experiments that we did in our model. The default model mentioned above will have an accuracy of 65\%\ in test data with 100 epochs. 
We also try to use Average Pooling instead of Max Pooling, but the accuracy of the model resulted to 63.78%. Hence, we decide to use the Max Pooling instead. 
From the experiments, we notice that the accuracy of training data can reach 90\%\ while test images remain at about 65%. 
It shows that we need to solve the overfitting problem. We add the dropout after the first two Linear models with the dropout rate of 0.5. 
This resulted in an increase in the accuracy of the model in test data as it reaches 74.15%. 

While that was an improvement, the issue of overfitting still exists. Then, we decided to increase the neurons in each layer but it did not affect the accuracy as it could not break 75% mark. 
After viewing several journals and aricles, we decide to add transforms to generate more training data while modifying the images. 
First, we use transforms.Compose([transforms.RandomHorizontalFlip(),transforms.RandomRotation(10),transforms.ToTensor(),
transforms.RandomErasing(p=0.75,scale=(0.02, 0.1),value=1.0, inplace=False)]) to set the train model. 
The accuracy increased to 77.38\%\ which is a slight increament in what we had before. Then, we change the RandomRotation from 10 to RandomRotation(20) and got an accuracy of 82\%\ after 250 epochs. 
We realised that the probability of RandomErasing is too high so, we decided to modify it to 0.5 and this resulted to an accuracy of 84\%\ after 250 epochs. 
After viewing the images, we realised that rotation of the image is not the best choice for the task. 
Instead, we choose to use transforms.RandomCrop(64, padding=4, padding_mode='reflect') to replace to RandomRotation which resulted in an accuracy of 82\%\ after 40 epochs. 
To increase the accuarcy, we increased the neurons in linear layer and it resulted to an accuracy of 85\%\ accuracy after 150 epochs.

Below is the table that records test.

Transform:              None        None        RandomHorizontalFlip, RandomRotation, RandomErasing       RandomCrop, RandomHorizontalFlip,Normalize            RandomCrop, RandomHorizontalFlip,Normalize          RandomCrop, RandomHorizontalFlip,Normalize           RandomCrop, RandomHorizontalFlip,Normalize
Num of Conv layer:      3           3           3                                                         2                                                     3                                                   4                                                    5
Num of Linear layer:    3           3           3                                                         1                                                     1                                                   2                                                    2
Max Pool:	            0           3           3                                                         1                                                     1                                                   2                                                    2
Avg Pool:               3           0           0                                                         0                                                     0                                                   1                                                    0
Dropout	Loss:           0           0.5         0.5                                                       0                                                     0                                                   0                                                    0
Function:               Yue_1       Yue_2       Yue_3                                                     musa_1                                                musa_2                                              musa_3                                              musa_4    
Num of Epoch:           40          100         200                                                       20                                                    20                                                  100                                                 100
Learning rate:          0.001       0.001       0.001                                                     0.001                                                 0.001                                               0.001                                               0.001
Optimizer:              Adam        Adam        Adam                                                      Adam                                                  Adam                                                Adam                                                Adam
train_val_split         0.8         0.8         0.8                                                       0.8                                                   0.8                                                 0.8                                                 0.8                                                                     
Accuracy:               64.78%      74%         82%                                                       55%                                                   69%                                                 81%                                                 85%


"""
#########################################
######    Experimental network     ######
#########################################
# Network used for testing. 
# Function: Yue_1
# Using average pooling instead of max pooling.
# get an accuracy in test imaged around 63.78% after 40 epochs


# def transform(mode):
#     """
#     Useing transforms to reducing overfitting
#     """
#     if mode == 'train':
#         return transforms.Compose([transforms.ToTensor()])
#     elif mode == 'test':
#         return transforms.Compose([transforms.ToTensor()])
# class Network(nn.Module):
#     def __init__(self):
#         super(Network, self).__init__()
#         # Building the convolutional neural network based on LeNet-5
#         self.conv1 = nn.Conv2d(3, 12, 5)
#         self.conv2 = nn.Conv2d(12, 48, 5)
#         self.conv3 = nn.Conv2d(48, 144, 5)
#         self.line1 = nn.Linear(2304,900)
#         self.line2 = nn.Linear(900,200)
#         self.line3 = nn.Linear(200,14)
#         self.soft = nn.LogSoftmax(dim=1)
#         self.rel = nn.ReLU()
#         self.avgP = nn.AvgPool2d(2)
        
#     def forward(self, t):
#         x = self.conv1(t)
#         x = self.rel(x)
#         x = self.avgP(x)
#         x = self.conv2(x)
#         x = self.rel(x)
#         x = self.avgP(x)
#         x = self.conv3(x)
#         x = self.rel(x)
#         x = self.avgP(x)
#         x = x.reshape(x.shape[0],-1)
#         x = self.line1(x)
#         x = self.rel(x)
#         x = self.line2(x)
#         x = self.rel(x)
#         x = self.line3(x)
#         return x

#########################################################################

# Function: Yue_2
# Adding dropout to linear layer
# get an accuracy in test imaged around 74% after 100 epochs


# def transform(mode):
#     """
#     Useing transforms to reducing overfitting
#     """
#     if mode == 'train':
#         return transforms.Compose([transforms.ToTensor()])
#     elif mode == 'test':
#         return transforms.Compose([transforms.ToTensor()])
# class Network(nn.Module):
#     def __init__(self):
#         super(Network, self).__init__()
#         # Building the convolutional neural network based on LeNet-5
#         self.conv1 = nn.Conv2d(3, 12, 5)
#         self.conv2 = nn.Conv2d(12, 48, 5)
#         self.conv3 = nn.Conv2d(48, 144, 5)
#         self.line1 = nn.Linear(2304,900)
#         self.line2 = nn.Linear(900,200)
#         self.line3 = nn.Linear(200,14)
#         self.soft = nn.LogSoftmax(dim=1)
#         self.rel = nn.ReLU()
#         self.avgP = nn.MaxPool2d(2)
#         self.drop1 = nn.Dropout(0.5)
        
#     def forward(self, t):
#         x = self.conv1(t)
#         x = self.rel(x)
#         x = self.avgP(x)
#         x = self.conv2(x)
#         x = self.rel(x)
#         x = self.avgP(x)
#         x = self.conv3(x)
#         x = self.rel(x)
#         x = self.avgP(x)
#         x = x.reshape(x.shape[0],-1)
#         x = self.line1(x)
#         x = self.drop1(x)
#         x = self.rel(x)
#         x = self.line2(x)
#         x = self.drop1(x)
#         x = self.rel(x)
#         x = self.line3(x)
#         return x

#########################################################################


# Function: Yue_3
# Adding RandomHorizontalFlip, RandomRotation, RandomErasing in data transform
# get an accuracy in test imaged around 82% after 200 epochs


# def transform(mode):
#     """
#     Useing transforms to reducing overfitting
#     """
#     if mode == 'train':
#         return transforms.Compose([transforms.RandomHorizontalFlip(), 
#           transforms.RandomRotation(20),transforms.ToTensor(),
#           transforms.RandomErasing(p=0.75,scale=(0.02, 0.1),value=1.0, inplace=False)])
#     elif mode == 'test':
#         return transforms.Compose([transforms.ToTensor()])


# class Network(nn.Module):
#     def __init__(self):
#         super(Network, self).__init__()
#         # Building the convolutional neural network based on LeNet-5
#         self.conv1 = nn.Conv2d(3, 12, 5)
#         self.conv2 = nn.Conv2d(12, 48, 5)
#         self.conv3 = nn.Conv2d(48, 144, 5)
#         self.line1 = nn.Linear(2304,900)
#         self.line2 = nn.Linear(900,200)
#         self.line3 = nn.Linear(200,14)
#         self.soft = nn.LogSoftmax(dim=1)
#         self.rel = nn.ReLU()
#         self.avgP = nn.MaxPool2d(2)
#         self.drop1 = nn.Dropout(0.5)
        
#     def forward(self, t):
#         x = self.conv1(t)
#         x = self.rel(x)
#         x = self.avgP(x)
#         x = self.conv2(x)
#         x = self.rel(x)
#         x = self.avgP(x)
#         x = self.conv3(x)
#         x = self.rel(x)
#         x = self.avgP(x)
#         x = x.reshape(x.shape[0],-1)
#         x = self.line1(x)
#         x = self.drop1(x)
#         x = self.rel(x)
#         x = self.line2(x)
#         x = self.drop1(x)
#         x = self.rel(x)
#         x = self.line3(x)
#         return x




# Network used for testing. 
# Function: musa_1

# class Network(nn.Module):
#     def __init__(self):
#         super(Network, self).__init__()
#         self.conv1 = nn.Sequential(        
#             nn.Conv2d(
#                 in_channels=1,             
#                 out_channels=10,           
#                 kernel_size=5,             
#                 stride=1,                  
#                 padding=2,                  
#             ),                              
#             nn.ReLU(),                     
#             nn.MaxPool2d(kernel_size=2),   
#         )
#         self.conv2 = nn.Sequential(         
#             nn.Conv2d(10, 32, 5, 1, 2),     
#             nn.ReLU(),                      
#             nn.MaxPool2d(2),                
#         )

#         self.rel = nn.ReLU()
#         self.out = nn.Linear(8192, 14)  

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = x.view(x.size(0), -1)           
#         output = self.out(x)
#         return output  






# Network used for testing. 
# Function: musa_2

# class Network(nn.Module):
#     def __init__(self):
#         super(Network, self).__init__()
#         self.conv1 = nn.Sequential(         
#             nn.Conv2d(
#                 in_channels=1,              
#                 out_channels=10,            
#                 kernel_size=5,             
#                 stride=1,                   
#                 padding=2,                  
#             ),                             
#             nn.ReLU(),                      
#             nn.MaxPool2d(kernel_size=2),    
#         )
#         self.conv2 = nn.Sequential(         
#             nn.Conv2d(10, 32, 5, 1, 2),     
#             nn.ReLU(),                     
#             nn.MaxPool2d(2),               
#         )
#         self.conv3 = nn.Sequential(       
#             nn.Conv2d(32, 64, 5, 1, 2),     
#             nn.ReLU(),                      
#             nn.MaxPool2d(2),                
#         )

#         self.rel = nn.ReLU()
#         self.out = nn.Linear(4096, 14)   

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = self.conv3(x)
#         x = x.view(x.size(0), -1)           
#         output = self.out(x)
#         return output  


# Network used for testing. 
# Function: musa_3


# class Network(nn.Module):
#     def __init__(self):
#         super(Network, self).__init__()
#         self.conv1 = nn.Sequential(        
#             nn.Conv2d(
#                 in_channels=1,             
#                 out_channels=10,           
#                 kernel_size=5,             
#                 stride=1,                   
#                 padding=2,                  
#             ),                             
#             nn.ReLU(),                      
#             nn.MaxPool2d(kernel_size=2),    
#         )
#         self.conv2 = nn.Sequential(         
#             nn.Conv2d(10, 32, 5, 1, 2),     
#             nn.ReLU(),                      
#             nn.MaxPool2d(2),               
#         )
#         self.conv3 = nn.Sequential(         
#             nn.Conv2d(32, 64, 5, 1, 2),     
#             nn.ReLU(),                      
#             nn.MaxPool2d(2),                
#         )

#         self.conv4 = nn.Sequential(        
#             nn.Conv2d(64, 256, 5, 1, 2),     
#             nn.ReLU(),                      
#             nn.MaxPool2d(2),                
#         )

#         self.rel = nn.ReLU()
#         self.hid_1 = nn.Linear(4096, 2000)
#         self.out = nn.Linear(2000, 14)   
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = self.conv3(x)
#         x = self.conv4(x)
#         x = x.view(x.size(0), -1)           
#         x = self.hid_1(x)
#         x = self.rel(x)
#         output = self.out(x)
#         return output   




# Network used for testing. 
# Function: musa_4


# def transform(mode):
#     """
#     Called when loading the data. Visit this URL for more information:
#     https://pytorch.org/vision/stable/transforms.html
#     You may specify different transforms for training and testing
#     """
#     if mode == 'train':
#         return transforms.Compose([transforms.Grayscale(1), transforms.RandomCrop(64, padding=4, padding_mode='reflect'), 
#                          transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.5, ), (0.5, ))])
#     elif mode == 'test':
#         return transforms.Compose([transforms.Grayscale(1), transforms.ToTensor(), transforms.Normalize((0.5, ), (0.5, ))])


# class Network(nn.Module):
#     def __init__(self):
#         super(Network, self).__init__()
#         self.conv1 = nn.Sequential(         
#             nn.Conv2d(
#                 in_channels=1,              
#                 out_channels=10,            
#                 kernel_size=5,              
#                 stride=1,                   
#                 padding=2,                  
#             ),                              
#             nn.ReLU(),                      
#             nn.MaxPool2d(kernel_size=2),    
#         )
#         self.conv2 = nn.Sequential(         
#             nn.Conv2d(10, 32, 5, 1, 2),     
#             nn.ReLU(),                     
#             nn.MaxPool2d(2),                
#         )
#         self.conv3 = nn.Sequential(         
#             nn.Conv2d(32, 64, 5, 1, 2),     
#             nn.ReLU(),                      
#             nn.MaxPool2d(2),                
#         )

#         self.conv4 = nn.Sequential(         
#             nn.Conv2d(64, 256, 5, 1, 2),    
#             nn.ReLU(),                     
#             nn.MaxPool2d(2),                
#         )

#         self.conv5 = nn.Sequential(         
#             nn.Conv2d(256, 1024, 5, 1, 2),     
#             nn.ReLU(),                      
#             nn.MaxPool2d(2),               
#         )
#         self.rel = nn.ReLU()
#         self.out = nn.Linear(4096, 14)   

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = self.conv3(x)
#         x = self.conv4(x)
#         x = self.conv5(x)
#         x = x.view(x.size(0), -1)           
#         output = self.out(x)
#         return output   




############################################################################
######     Specify transform(s) to be applied to the input images     ######
############################################################################
def transform(mode):
    """
    Called when loading the data. Visit this URL for more information:
    https://pytorch.org/vision/stable/transforms.html
    You may specify different transforms for training and testing
    """
    """
    Useing transforms to reducing overfitting
    """
    if mode == 'train':
        return transforms.Compose([transforms.RandomHorizontalFlip(),transforms.RandomCrop(64, padding=4, padding_mode='reflect'),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.5,scale=(0.02, 0.12),value=1, inplace=False)])
    elif mode == 'test':
        return transforms.Compose([transforms.ToTensor()])

############################################################################
######   Define the Module to process the images and produce labels   ######
############################################################################
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        # Building the convolutional neural network based on LeNet-5
        self.conv1 = nn.Conv2d(3, 12, 5)
        self.conv2 = nn.Conv2d(12, 42, 5)
        self.conv3 = nn.Conv2d(42, 250, 5)
        self.line1 = nn.Linear(4000,600)
        self.line2 = nn.Linear(600,84)
        self.line3 = nn.Linear(84,14)
        self.soft = nn.LogSoftmax(dim=1)
        self.rel = nn.ReLU()
        self.maxP1 = nn.MaxPool2d(2)
        self.drop1 = nn.Dropout(0.5)
        
    def forward(self, t):
        x = self.conv1(t)
        x = self.rel(x)
        x = self.maxP1(x)
        x = self.conv2(x)
        x = self.rel(x)
        x = self.maxP1(x)
        x = self.conv3(x)
        x = self.rel(x)
        x = self.maxP1(x)
        x = x.reshape(x.shape[0],-1)
        x = self.line1(x)
        x = self.drop1(x)
        x = self.rel(x)
        x = self.line2(x)
        x = self.drop1(x)
        x = self.rel(x)
        x = self.line3(x)
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
        pass


net = Network()
# choose cross entropy loass function
lossFunc = nn.CrossEntropyLoss()
############################################################################
#######              Metaparameters and training options              ######
############################################################################
dataset = "./data"
train_val_split = 0.8
batch_size = 100
epochs = 150
optimiser = optim.Adam(net.parameters(), lr=0.001)
