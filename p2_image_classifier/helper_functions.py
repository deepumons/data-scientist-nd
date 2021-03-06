import torch
from torch import nn, optim
from torchvision import transforms, datasets, models

from PIL import Image

def get_dataloaders(data_dir):

    # Set the name of the data directories for each type of data loader
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # Define your transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    # Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=test_transforms)

    # Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=50, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=50)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=50)

    class_to_idx = train_data.class_to_idx

    return trainloader, testloader, validloader, class_to_idx

def get_model(architecture, input_units, hidden_units, learning_rate):

    model = None

    if architecture == "alexnet":
        model = models.alexnet(pretrained = True)
    elif architecture == "vgg19":
        model = models.vgg19(pretrained = True)
    elif architecture == "densenet121":
        model = models.densenet121(pretrained = True)
    else:
        model = None

    # Disable backprogagation on model parameters
    for param in model.parameters():
        param.requires_grad = False

    # Set up the Classifier
    classifier = nn.Sequential(nn.Linear(input_units, hidden_units),
                            nn.ReLU(),
                            nn.Dropout(p=0.2),
                            nn.Linear(hidden_units, 102),
                            nn.LogSoftmax(dim=1))
    model.classifier = classifier

    return model

# Definition for the training function
def train_model(trainloader, model, learning_rate, device):

    # Channge model to train mode
    model.train()

    training_loss = 0

    # Define the loss function
    criterion = nn.NLLLoss()

    # Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    # move the model to the default device
    model.to(device);

    for i, (inputs, labels) in enumerate(trainloader):

        # Clear the gradients
        optimizer.zero_grad()

        # move the inputs and labels to the default device
        inputs = inputs.to(device)
        labels = labels.to(device)

        #Make a forward pass on the network and compute the loss
        outputs = model.forward(inputs)
        loss = criterion(outputs, labels)
        # Back propagation step
        loss.backward()
        # Update the weights
        optimizer.step()

        training_loss += loss.item()

    print(f"Training loss: {training_loss/len(trainloader):.3f}", end=" ")

# Define the testing function
def test_model(testloader, model, device):

    # Change model to evaluation mode
    model.eval()

    # Define the loss function
    criterion = nn.NLLLoss()

    testing_loss = 0
    accuracy = 0

    # disable auto grad
    with torch.no_grad():

        for i, (inputs, labels) in enumerate(testloader):

            inputs = inputs.to(device)
            labels = labels.to(device)

            log_ps = model.forward(inputs)
            loss = criterion(log_ps, labels)

            testing_loss += loss.item()

            # Calculate accuracy
            ps = torch.exp(log_ps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()


    print(f"Testing loss: {testing_loss/len(testloader):.3f}", end=" ")
    print(f"Accuracy: {accuracy/len(testloader):.3f}", end=" ")

    # Write a function that loads a checkpoint and rebuilds the model
def load_checkpoint(filepath):
    # Load the checkpoint
    checkpoint = torch.load(filepath, map_location={'cuda:0': 'cpu'})

    # Retrieve the params from the model
    architecture = checkpoint['architecture']

    # Recreate the model
    model = None
    input_units = 0

    if architecture == "alexnet":
        model = models.alexnet(pretrained = True)
        input_units = 9216
    elif architecture == "vgg19":
        model = models.vgg19(pretrained = True)
        input_units = 25088
    elif architecture == "densenet121":
        model = models.densenet121(pretrained = True)
        input_units = 1024

    hidden_units = checkpoint['hidden_units']

    # Set up the Classifier
    classifier = nn.Sequential(nn.Linear(input_units, hidden_units),
                            nn.ReLU(),
                            nn.Dropout(p=0.2),
                            nn.Linear(hidden_units, 102),
                            nn.LogSoftmax(dim=1))
    model.classifier = classifier

    # load the remaining items from the checkpoint and attach to the model
    model.load_state_dict(checkpoint['model_state_dict'], strict = False)
    model.class_to_idx = checkpoint['class_to_idx']

    #print(f">architecture:{architecture}")

    # Disable backprogagation on model parameters
    for param in model.parameters():
        param.requires_grad = False

    return model

def process_image(image_path):

    # load the image from the file system
    single_image = Image.open(image_path)

    # define the transformations to be applied on the input image
    image_transforms = transforms.Compose([transforms.Resize(256),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

    # apply the transforms to produce an image tensor
    image_tensor = image_transforms(single_image)

    return image_tensor


def predict(image_path, model, topk):

    # process the image and get the image tensor
    image = process_image(image_path)

    # add and additional dim before passing to the model
    image = image.unsqueeze_(0)
    image = image.float()

    # change the model to evaluation mode and move to CPU
    model.eval()
    #model.to('cpu')

    # make a forward pass of the model and compute the top probablities, classes
    with torch.no_grad():
        log_ps = model.forward(image)

    ps = torch.exp(log_ps)

    return ps.topk(topk, dim=1)
