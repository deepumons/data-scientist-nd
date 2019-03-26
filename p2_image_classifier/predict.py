import argparse
import torch
import json
import time

import numpy as np

#from PIL import Image
from helper_functions import get_dataloaders, test_model, load_checkpoint, predict

parser = argparse.ArgumentParser(description='Trainer for Image Classifier project.')

# Required arguments
parser.add_argument('path_to_image',
                    action='store',
                    default='flowers',
                    help='Path to the image to be classifed including file name and extension.')

parser.add_argument('checkpoint',
                    action='store',
                    default='checkpoint.pth',
                    help='Name of the checkpoint (Assumes its present in the same folder.)')

# Optional arguments
parser.add_argument('--top_k',
                    default=1,
                    action='store',
                    type=int,
                    help='Number of prediction probabablities to be returned.')

parser.add_argument('--gpu',
                    default=False,
                    action='store_true',
                    help='Boolean flag to use GPU for training')

parser.add_argument('--category_names',
                    default="cat_to_name.json",
                    action='store',
                    help='Name of the chekpoint (Assumes its present in the same folder.)')

# Parse the input arguments
args = parser.parse_args()
print(args)

# Save arguments as global parameters
path_to_image = args.path_to_image
checkpoint = args.checkpoint
top_k = args.top_k
use_gpu = args.gpu
category_names = args.category_names

# Set the default device for training
device = torch.device("cpu")
if (use_gpu):
    if not torch.cuda.is_available():
        print("No GPU is available on this system. Please disable the --gpu flag and try again.")
        exit()
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the id to name mapping for flowers
with open(category_names, 'r') as f:
    cat_to_name = json.load(f)

# Save the model for later use
print()
print("Trying to load the trained model from the checkpoint...")
model = load_checkpoint(checkpoint)
print("Successfully loaded the model from the checkpoint.")

# The following code can be executed if testing is requierd
do_test = False
if (do_test):
    # Load the test data
    print("Preparing data loaders...")
    _, testloader, _, _ = get_dataloaders("flowers")
    print("Finished preparing data loaders.")

    # Test the model wit the test data
    test_start = time.time()
    test_model(testloader, model, device)
    print(f"[Time: {(time.time() - test_start)/60:.2f} mins]", end="")


# Make predictions
test_image_index = path_to_image.split('/')[2]
test_image_name = cat_to_name[test_image_index]
#print(test_image_index)
print(f"The flower image supplied by you is: {test_image_name}")

probs, classes = predict(path_to_image, model, top_k)
#print(probs)
#print(classes)

# Print the top classes
top_classes = np.array(classes)[0]

# interchange keys and values in class_to_idx so that it's easily searchable
class_to_idx = model.class_to_idx
idx_to_class = {}

for keys, values in class_to_idx.items():
    idx_to_class[values] = keys

# list to store the top flower names
flower_names = list()

for i in top_classes:
    flower_names.append(cat_to_name[str(idx_to_class[i])])

print(f"Top {top_k} predictions for this image (in decreasing order of probablity) are: ")
print(flower_names)
