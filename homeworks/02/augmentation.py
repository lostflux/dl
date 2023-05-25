#!/usr/bin/env python3
# -*- coding: utf-8 -*-


## Importing all libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import OrderedDict
import seaborn as sb
import cv2
import json
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import sys


##Load Data from paths
##Update data_dir path to your flowers directory location
#************************* Write your code here *********************

data_dir = "dataset"


##************************* Your code ends here***********************
train_dir = f"{data_dir}/flowers/train"
valid_dir = f"{data_dir}/flowers/valid"
test_dir = f"{data_dir}/flowers/test"

##Mapping the labels
##Update path to your flower_to_name.json file
#************************* Write your code here *********************

with open(f'{data_dir}/flower_to_name.json', 'r') as f:
	flower_to_name = json.load(f)
	print(f"flower_to_name: {flower_to_name}")


##************************* Your code ends here***********************


## Define transforms for the training, validation, and testing sets
##Tech0/1/2/3 training_transforms:

#************************* Write your code here *********************

training_transforms = transforms.Compose([
	transforms.RandomCrop(size=32, padding=4),
	transforms.RandomHorizontalFlip(p=0.5)
])

##************************* Your code ends here***********************


validation_transforms = transforms.Compose([transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], 
                                                                 [0.229, 0.224, 0.225])])

testing_transforms = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], 
                                                              [0.229, 0.224, 0.225])])

#Load the datasets with ImageFolder
training_dataset = datasets.ImageFolder(train_dir, transform=training_transforms)
validation_dataset = datasets.ImageFolder(valid_dir, transform=validation_transforms)
testing_dataset = datasets.ImageFolder(test_dir, transform=testing_transforms)

#Using the image datasets and the trainforms, define the dataloaders
train_loader = torch.utils.data.DataLoader(training_dataset, batch_size=128, shuffle=True, workers=4)
validate_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=32)
test_loader = torch.utils.data.DataLoader(testing_dataset, batch_size=100)

model = models.resnet18(pretrained=True)
# Freeze pretrained model parameters to avoid backpropogating through them
for parameter in model.parameters():
    parameter.requires_grad = True


# Build custom classifier
classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(512,64)),
                                        ('relu', nn.ReLU()),
                                        ('drop', nn.Dropout(p=0.5)),
                                        ('fc2', nn.Linear(64, 20)),
                                        ('output', nn.LogSoftmax(dim=1))]))
model.fc = classifier

# Function for the validation pass
def validation(model, validateloader, criterion):
    
	val_loss = 0
	accuracy = 0
	
	for images, labels in iter(validateloader):
		device = 'cuda' if torch.cuda.is_available() else 'cpu'
		images, labels = images.to(device), labels.to(device)

		output = model.forward(images)
		val_loss += criterion(output, labels).item()

		probabilities = torch.exp(output)
		
		equality = (labels.data == probabilities.max(dim=1)[1])
		accuracy += equality.type(torch.FloatTensor).mean()
	
	return val_loss, accuracy

# Loss function and gradient descent
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_classifier():
#************************* Write your code here *********************

	epochs = 0

##************************* Your code ends here***********************
	steps = 0
	print_every = 5
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	#model.to('cuda')
	model.to(device)

	for e in range(epochs):
		
		model.train()
		running_loss = 0

		for images, labels in iter(train_loader):
			steps += 1
			images, labels = images.to(device), labels.to(device)
			optimizer.zero_grad()

			output = model.forward(images)
			loss = criterion(output, labels)
			loss.backward()
			optimizer.step()

			running_loss += loss.item()
			#print(steps)
			if steps % print_every == 0:
				model.eval()
			
				# Turn off gradients for validation, saves memory and computations
				with torch.no_grad():
					validation_loss, accuracy = validation(model, validate_loader, criterion)

		
		print("Epoch: {}/{}.. ".format(e+1, epochs),
						"Training Loss: {:.3f}.. ".format(running_loss/print_every),
						"Validation Loss: {:.3f}.. ".format(validation_loss/len(validate_loader)))

		running_loss = 0
		model.train()
                    
train_classifier()    

def test_accuracy(model, test_loader):

	# Do validation on the test set
	model.eval()
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	#model.to('cuda')
	model.to(device)
	with torch.no_grad():
		accuracy = 0

		for images, labels in iter(test_loader):
			device = 'cuda' if torch.cuda.is_available() else 'cpu'
			images, labels = images.to(device), labels.to(device)
			output = model.forward(images)
			probabilities = torch.exp(output)
	
			equality = (labels.data == probabilities.max(dim=1)[1])
			accuracy += equality.type(torch.FloatTensor).mean()
		
		print("Test Accuracy: {}".format(accuracy/len(test_loader)))    
        
        
test_accuracy(model, test_loader)

##Save your models trained for 50 epochs
## Save your test accuracies changing epochs and augmentations Tech0/1/2/3
## Plot the line graph
#************************* Write your code here *********************


##************************* Your code ends here***********************
