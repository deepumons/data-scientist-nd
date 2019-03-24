import argparse
import helperfunctions

import torch
from torch import nn, optim
from torchvision import transforms, datasets, models

import time
import numpy as np

from PIL import Image

parser = argparse.ArgumentParser(description='Trainer for Image Classifier project.')

# required argument
parser.add_argument('data_directory',
                    action='store',
                    default='flowers',
                    help='Location of the folder with images for training, testing and validation.')

parser.add_argument('--save_dir',
                    action='store',
                    dest='save_directory',
                    help='Target location to save the model checkpoint after training.')

parser.add_argument('--arch',
                    action='store',
                    default='vgg19',
                    dest='architecture',
                    help = 'Name of the pre-trained model to be used. The default is vgg19.')

parser.add_argument('--learning_rate',
                    type=int,
                    action='store',
                    default=0.001,
                    help='The learning rate for the optimizer. The Default is 0.001.')

parser.add_argument('--hidden_units',
                    type=int,
                    action='store',
                    default=256,
                    help='Number of hidden units to be used. The default is 256.')

parser.add_argument('--epochs',
                    type=int,
                    action='store',
                    default=5,
                    help='Number of epochs. The default is 5')

# Parse the input arguments
args = parser.parse_args()
#print(args)

# Save arguments as global parameters
data_directory = args.data_directory
save_directory = args.save_directory
architecture = args.architecture
learning_rate = args.learning_rate
hidden_units = args.hidden_units
epochs = args.epochs

# Load the training, testing and validation data from data directroy
trainloader, testloader, validloader = get_dataloaders(data_directory)

# initialize the model
model = get_model(architecture, hidden_units, learning_rate)

# Use GPU if it's available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# move the model to the default device
model.to(device);

# Define the loss function
criterion = nn.NLLLoss()

# Only train the classifier parameters, feature parameters are frozen
optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

# to be removed
epochs = 1

# Training the model
for e in range(epochs):

    train_start = time.time()

    print()
    print(f"Epoch {e+1}", end=" ")
    train_model(trainloader, model, criterion, optimizer)
    print(f"[Time: {(time.time() - train_start)/60:.2f} mins]", end=" ")

    validation_start = time.time()

    test_model(validloader, model, criterion)
    print(f"[Time: {(time.time() - validation_start)/60:.2f} mins]", end="")
