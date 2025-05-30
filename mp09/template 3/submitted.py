# submitted.py

"""
This is the module you will submit to the autograder.

There are several function and variable definitions, here, that raise RuntimeErrors.
You should replace each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.

If you are not sure how to use PyTorch, you may want to take a look at the tutorial.
"""

import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from models import resnet18


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


"""
1.  Define and build a PyTorch Dataset
"""
class CIFAR10(Dataset):
    def __init__(self, data_files, transform=None, target_transform=None):
        """
        Initialize your dataset here. Note that transform and target_transform
        correspond to your data transformations for train and test respectively.
        """
        #raise NotImplementedError("You need to write this part!")
        self.images = []
        self.labels = []
        for file in data_files:
            temp = unpickle(file)
            self.images += temp[b'data']
            self.labels += temp[b'labels']
        for i in range(0, len(self.images)):
            temp_reshape = np.reshape(self.images[i], (3, 32, 32))
            self.images[i] = np.transpose(temp_reshape, (1, 2, 0))
        self.transform = transform
        self.target_transform = target_transform

        
    def __len__(self):
        """
        Return the length of your dataset here.
        """
        #raise NotImplementedError("You need to write this part!")
        return len(self.images)
    

    def __getitem__(self, idx):
        """
        Obtain a sample from your dataset. 

        Parameters:
            x:      an integer, used to index into your data.

        Outputs:
            y:      a tuple (image, label), although this is arbitrary so you can use whatever you would like.
        """
        #raise NotImplementedError("You need to write this part!")
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    

def get_preprocess_transform(mode):
    """
    Parameters:
        mode:           "train" or "test" mode to obtain the corresponding transform
    Outputs:
        transform:      a torchvision transforms object e.g. transforms.Compose([...]) etc.
    """

    #raise NotImplementedError("You need to write this part!")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    return transform


def build_dataset(data_files, transform=None):
    """
    Parameters:
        data_files:      a list of strings e.g. "cifar10_batches/data_batch_1" corresponding to the CIFAR10 files to load data
        transform:       the preprocessing transform to be used when loading a dataset sample
    Outputs:
        dataset:      a PyTorch dataset object to be used in training/testing
    """
    #raise NotImplementedError("You need to write this part!")
    dataset = CIFAR10(data_files = data_files, transform = transform)
    return dataset


"""
2.  Build a PyTorch DataLoader
"""
def build_dataloader(dataset, loader_params):
    """
    Parameters:
        dataset:         a PyTorch dataset to load data
        loader_params:   a dict containing all the parameters for the loader. 
        
    Please ensure that loader_params contains the keys "batch_size" and "shuffle" corresponding to those 
    respective parameters in the PyTorch DataLoader class. 

    Outputs:
        dataloader:      a PyTorch dataloader object to be used in training/testing
    """
    #raise NotImplementedError("You need to write this part!")
    dataloader = DataLoader(dataset, **loader_params)
    return dataloader


"""
3. (a) Build a neural network class.
"""
class FinetuneNet(torch.nn.Module):
    def __init__(self):
        """
        Initialize your neural network here. Remember that you will be performing finetuning
        in this network so follow these steps:
        
        1. Initialize convolutional backbone with pretrained model parameters.
        2. Freeze convolutional backbone.
        3. Initialize linear layer(s). 
        """
        super().__init__()
        ################# Your Code Starts Here #################
        self.model = resnet18()
        self.model.load_state_dict(torch.load('resnet18.pt'))
        #raise NotImplementedError("You need to write this part!")
        ################## Your Code Ends here ##################

    def forward(self, x):
        """
        Perform a forward pass through your neural net.

        Parameters:
            x:      an (N, input_size) tensor, where N is arbitrary.

        Outputs:
            y:      an (N, output_size) tensor of output from the network
        """
        ################# Your Code Starts Here #################
        y = self.model(x)
        return y
        #raise NotImplementedError("You need to write this part!")
        ################## Your Code Ends here ##################


"""
3. (b)  Build a model
"""
def build_model(trained=False):
    """
    Parameters:
        trained:         a bool value specifying whether to use a model checkpoint

    Outputs:
        model:           the model to be used for training/testing
    """
    net = FinetuneNet()
    return net


"""
4.  Build a PyTorch optimizer
"""
def build_optimizer(optim_type, model_params, hparams):
    """
    Parameters:
        optim_type:      the optimizer type e.g. "Adam" or "SGD"
        model_params:    the model parameters to be optimized
        hparams:         the hyperparameters (dict type) for usage with learning rate 

    Outputs:
        optimizer:       a PyTorch optimizer object to be used in training
    """
    #raise NotImplementedError("You need to write this part!")
    if optim_type == "Adam":
        optimizer = torch.optim.Adam(params = model_params, **hparams)
    else:
        optimizer = torch.optim.SGD(params = model_params, **hparams)
    return optimizer


"""
5. Training loop for model
"""
def train(train_dataloader, model, loss_fn, optimizer):
    """
    Train your neural network.

    Iterate over all the batches in dataloader:
        1.  The model makes a prediction.
        2.  Calculate the error in the prediction (loss).
        3.  Zero the gradients of the optimizer.
        4.  Perform backpropagation on the loss.
        5.  Step the optimizer.

    Parameters:
        train_dataloader:   a dataloader for the training set and labels
        model:              the model to be trained
        loss_fn:            loss function
        optimizer:          optimizer
    """

    ################# Your Code Starts Here #################
    if torch.cuda.is_available():
        cpuORgpu = "cuda"
    else:
        cpuORgpu = "cpu"
    device = torch.device(cpuORgpu)
    model = model.to(device)

    for features, labels in train_dataloader: # features -> images
        features, labels = features.to(device), labels.to(device)
        labels_pred = model.forward(features)
        loss = loss_fn(labels_pred, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    #raise NotImplementedError("You need to write this part!")
    ################## Your Code Ends here ##################


"""
6. Testing loop for model
"""
def test(test_dataloader, model):
    """
    This part is optional.

    You can write this part to monitor your model training process.

    Test your neural network.
        1.  Make sure gradient tracking is off, since testing set should only
            reflect the accuracy of your model and should not update your model.
        2.  The model makes a prediction.
        3.  Calculate the error in the prediction (loss).
        4.  Print the loss.

    Parameters:
        test_dataloader:    a dataloader for the testing set and labels
        model:              the model that you will use to make predictions


    Outputs:
        test_acc:           the output test accuracy (0.0 <= acc <= 1.0)
    """
    
    if torch.cuda.is_available():
        cpuORgpu = "cuda"
    else:
        cpuORgpu = "cpu"
    device = torch.device(cpuORgpu)
    model = model.to(device)

    total = 0
    accurate = 0
    test_acc = 0
    for features, labels in test_dataloader:
        with torch.no_grad():
            features, labels = features.to(device), labels.to(device)
            labels_pred = model.forward(features)
            for i in range(0, len(labels_pred)):
                total += 1
                if torch.argmax(labels_pred[i]) == labels[i]:
                    accurate += 1
    test_acc = accurate / total
    # test_loss = something
    # print("Test loss:", test_loss)
    #print("Testacc:", test_acc)
    return test_acc
    #raise NotImplementedError("You need to write this part!")

"""
7. Full model training and testing
"""
def run_model():
    """
    The autograder will call this function and measure the accuracy of the returned model.
    Make sure you understand what this function does.
    Do not modify the signature of this function (names and parameters).

    Please run your full model training and testing within this function.

    Outputs:
        model:              trained model
    """
    #raise NotImplementedError("You need to write this part!")
    data_files = [
        "cifar10_batches/data_batch_1", "cifar10_batches/data_batch_2", "cifar10_batches/data_batch_3", "cifar10_batches/data_batch_4", "cifar10_batches/data_batch_5",
    ]
    data_files1 = ["cifar10_batches/data_batch_4"]
    dataset = build_dataset(data_files1, transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ]))

    loader_params = {
        'batch_size': 64,
        'shuffle': True
    }
    dataloader = build_dataloader(dataset, loader_params = loader_params)

    model = build_model()
    hparams_optim = {
        'lr': 0.0001,
        'weight_decay': 0.01
    }
    optimizer = build_optimizer("Adam", model.parameters(), hparams = hparams_optim)
    loss_fn = torch.nn.CrossEntropyLoss()
    train(dataloader, model, loss_fn, optimizer)

    test_acc = test(dataloader, model)
    print ("Test accuracy: ", test_acc)

    return model

    
