import argparse
from helper_functions import get_dataloaders, get_model, train_model, test_model

import torch


import time
import numpy as np

from PIL import Image

parser = argparse.ArgumentParser(description='Trainer for Image Classifier project.')

# required argument
parser.add_argument('data_directory',
                    action='store',
                    default='flowers',
                    help='Location of the folder with images for training, testing and validation.')

# optional arguments
parser.add_argument('--save_dir',
                    default=".",
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

parser.add_argument('--gpu',
                    default=False,
                    action='store_true',
                    help='Boolean flag to use GPU for training')

# Parse the input arguments
args = parser.parse_args()

# Save arguments as global parameters
data_directory = args.data_directory
save_directory = args.save_directory
architecture = args.architecture
learning_rate = args.learning_rate
hidden_units = args.hidden_units
epochs = args.epochs
use_gpu = args.gpu

# Perform input sanity checks
allowed_architecture = ['alexnet', 'vgg19', 'densenet121']

# Check if user supplied architecture is present in the allowed architectures
if (architecture not in allowed_architecture):
    print(f"The architecture {architecture} is not supported. Please select alexnet, vgg19, or densenet121")
    exit()

# Set input input_units based on the architecture
input_units = 0

if architecture == "alexnet":
    input_units = 9216
elif architecture == "vgg19":
    input_units = 25088
elif architecture == "densenet121":
    input_units = 1024

# Check if user selected hidden_units are within range
if(hidden_units <= 102 or hidden_units >= input_units):
    print(f"Hidden units must fall between {input_units} and 102")
    exit()

# Set the default device for training
device = None
if (use_gpu):
    if not torch.cuda.is_available():
        print("No GPU is available on this system. Please disable the --gpu flag and try again.")
        exit()

# Set the default device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Inform the user about the parameters
print("Beginning Neural Network training with the following arguments:")
print(args)

# Load the training, testing and validation data from data directroy
print("Preparing data loaders...")
trainloader, testloader, validloader, class_to_idx = get_dataloaders(data_directory)
print("Finished preparing data loaders.")

# initialize the model
print(f"Downloading and preparing {architecture} model...")
model = get_model(architecture, input_units, hidden_units, learning_rate)
# Save the mapping of classes to indices to the model
model.class_to_idx = class_to_idx
print(f"Model {architecture} is ready.")

print(f"Beginning model training run for {epochs} epochs...")

# Training the model
for e in range(epochs):

    train_start = time.time()

    print()
    print(f"Epoch {e+1}", end=" ")
    train_model(trainloader, model, learning_rate, device)
    print(f"[Time: {(time.time() - train_start)/60:.2f} mins]", end=" ")

    validation_start = time.time()

    test_model(validloader, model, device)
    print(f"[Time: {(time.time() - validation_start)/60:.2f} mins]", end="")

# Save the model for later used
print()
print("Trying to save the trained model...")

file_name = "checkpoint_" + architecture + "_" + str(hidden_units) + "_" + str(epochs) + ".pth"
save_location = save_directory + "/" + file_name

checkpoint = {'architecture': architecture,
                'epoch_count' : epochs,
                'hidden_units' : hidden_units,
                'class_to_idx': model.class_to_idx,
                'model_state_dict': model.state_dict()}

torch.save(checkpoint, save_location)
print(f"Successfully saved the trained model to {save_location}")
