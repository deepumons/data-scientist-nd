# Data Scientist Nanodegree
# Deep Learning
## Project: Image Classifier

### Install

This project requires **Python 3.x** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [PyTorch](https://pytorch.org/get-started/locally/)

You will also need to have software installed to run and execute an [iPython Notebook](http://ipython.org/notebook.html)

We recommend students install [Anaconda](https://www.continuum.io/downloads), a pre-packaged Python distribution that contains all of the necessary libraries and software for this project.

### Code

The jupyter notebook named `Image Classifier Project.ipynb` has the step-by-step working code used to implement a deep neuaral network. Executing the cells in this notebook file in a top-down fashing should yield the expected result of training a neural network, and then using the same network to make predictions on new images.

This code is then ported to a console application with the help of python scripts that does the same operations in the jupyter notebook version. The details of these python scripts are listed below:
- train.py				> This file can be used to train the neural network using one of the pre-trained models.
- predict.py			> Once training is completed a checkpoint file is generated in the same folder. This can be supplied to predict.py along with a test image to make predictions on that image.
- helper_functions.py	> This file contains several utility, and dependent functions that is referenced by both train.py and predict.py and must be present in the same folder.

### Data

The project uses the training, testing, and validation image datasets of the [102 Category Flower Dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) by the Visual Geometry Group.

The images should be extracted to a /flowers local folder in the same directory as the jupyter notebook and python script files in order for them to work.

### Run

In a terminal or command window, navigate to the top-level project directory (that contains this README) and run one of the following commands:

```bash
ipython notebook Image Classifier Project.ipynb
```  
or
```bash
jupyter notebook Image Classifier Project.ipynb
```

This will open the iPython Notebook software and project file in your browser.

Executing the python scripts is straight forward using the following command structuretre:

python [train.py] | [predict.py] [mandatory parameters] [optional parameters]

For example:

python train.py "flowers" --gpu

python predict.py "flowers/test/28/image_05214.jpg" "checkpoint_vgg19_256_7.pth" --gpu

Please use the -h flag to get more information on all available parameters for the script files.

