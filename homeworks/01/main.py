import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import copy
import bound
import argparse
from torchvision import transforms, datasets

# train the model for one epoch on the given dataset
def train(model, device, train_loader, criterion, optimizer, epoch):
    sum_loss, sum_correct = 0, 0

    # switch to train mode
    model.train()

    for i, (data, target) in enumerate(train_loader):
        data, target = data.to(device).view(data.size(0),-1), target.to(device)

        # compute the output
        output = model(data)

        # compute the classification error and loss
        loss = criterion(output, target)
        pred = output.max(1)[1]
        sum_correct += pred.eq(target).sum().item()
        sum_loss += len(data) * loss.item()

        # compute the gradient and do an SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return 1 - (sum_correct / len(train_loader.dataset)), sum_loss / len(train_loader.dataset)


# evaluate the model on the given set
def validate(model, device, val_loader, criterion):
    sum_loss, sum_correct = 0, 0
    margin = torch.Tensor([]).to(device)

    # switch to evaluation mode
    model.eval()
    with torch.no_grad():
        for i, (data, target) in enumerate(val_loader):
            data, target = data.to(device).view(data.size(0), -1), target.to(device)

            # compute the output
            output = model(data)

            # compute the classification error and loss
            pred = output.max(1)[1]
            sum_correct += pred.eq(target).sum().item()
            sum_loss += len(data) * criterion(output, target).item()

            # compute the margin
            output_m = output.clone()
            for i in range(target.size(0)):
                output_m[i, target[i]] = output_m[i,:].min()

            #****************You will compute the margin below of your network*****************    
            margin = 

            #*************Your code ends here*************************

        #****************Return 5th percentile below********************
        percentile_margin = 

        #*************Your code ends here*************************
    return 1 - (sum_correct / len(val_loader.dataset)), sum_loss / len(val_loader.dataset), percentile_margin


# Load and Preprocess CIFAR-10 data.
# Loading: Dataset will be downloaded if the dataset is not in the given directory
# Preprocessing: You have to do normalizing each channel and data augmentation by random cropping and horizontal flipping
def load_data(split, dataset_name, datadir, nchannels):

    ##Normalize dataset
    ##mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    ##*************Write the code below to Normalize and do data augmentation*******************
    normalize = 
    tr_transform = 
    val_transform = 

    #*************Your code ends here*************************

    ##Load the datasets with the pre-processing
    get_dataset = getattr(datasets, dataset_name)
    
    if split == 'train':
        dataset = get_dataset(root=datadir, train=True, download=True, transform=tr_transform)
    else:
        dataset = get_dataset(root=datadir, train=False, download=True, transform=val_transform)

    return dataset


# This function trains a FC neural net with a singler hidden layer on the given dataset and calculates
# various measures on the learned network
# You have to write the parameters according to instructions, define the network and train it
def main():

    # Settings
    # Define the parameters to train your model
    #***********Check the instructions and make sure you set your parameters accordingly*********
    datadir     = "datasets"
    dataset     = "CIFAR10"     # dataset (other options: MNIST, CIFAR10, CIFAR100)
    nunits      = 1024          # hidden units
    lr          = 0.001         # learning rate
    mt          = 0.9           # momentum
    batchsize   = 64            # batch size
    epochs      = 25            # number of training epochs
    stopcond    = 0.01          # stop early if validation error goes below this threshold

    #*************Your code ends here*************************

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if device else {}
    nchannels, nclasses = 3, 10
    if (dataset == 'MNIST'): 
      nchannels = 1
    if (dataset == 'CIFAR100'): 
      nclasses = 100

    # create an initial model
    #*****************Define your 2-layer model here*****************
    # model = nn.Sequential()
    # model.append(nn.Linear(32*32*3, 1024))
    # model.append(nn.ReLU())
    # model.append(nn.Linear(1024, 10))

    # define model (32 x 32 x 3) -> (1024) -> (10)
    model = nn.Sequential(
        nn.Linear(32*32*3, 1024),
        nn.ReLU(),
        nn.Linear(1024, 10)
    )

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr, momentum=mt)

    # add optimizer to model
    model.append(optimizer)

    #*************Your code ends here********************************

    model = model.to(device)

    # create a copy of the initial model to be used later
    init_model = copy.deepcopy(model)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr, momentum=mt)

    # loading data
    train_dataset = load_data('train', dataset, datadir, nchannels)
    val_dataset = load_data('val', dataset, datadir, nchannels)

    train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True, **kwargs)
    val_loader = DataLoader(val_dataset, batch_size=batchsize, shuffle=False, **kwargs)

    # training the model
    for epoch in range(0, epochs):
        # *************You have to complete the code below for training your model****************

        tr_err, tr_loss = 
        #Validation run
        val_err, val_loss, val_margin = 

       #*************Your code ends here********************************

        print(f'Epoch: {epoch + 1}/{epochs}\t Training loss: {tr_loss:.3f}\t Validation margin {val_margin:.3f}\t ',
                f'Training error: {tr_err:.3f}\t Validation error: {val_err:.3f}')

        # stop training if the cross-entropy loss is less than the stopping condition
        if tr_loss < stopcond: 
            break
    # calculate the training error and margin (on Training set) of the learned model
    tr_err, tr_loss, tr_margin = validate( model, device, train_loader, criterion)
    print(f'\nFinal: Training loss: {tr_loss:.3f}\t Training margin {tr_margin:.3f}\t ',
            f'Training error: {tr_err:.3f}\t Validation error: {val_err:.3f}\n')
    # Print the measures and bounds computed in bound.py
    measure = bound.calculate(model, init_model, device, train_loader, tr_margin)
    for key, value in measure.items():
        print(f'{key:s}:\t {float(value):3.3}')
    

if __name__ == '__main__':
    main()
