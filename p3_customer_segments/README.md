# Data Scientist Nanodegree
# Unsupervised Learning
## Project: Identify Customer Segments with Arvato

### Install

This project requires **Python 3.x** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org)
- [matplotlib](http://matplotlib.org/)
- [seaborn](http://seaborn.pydata.org/)
- [scikit-learn](http://scikit-learn.org/stable/)

You will also need to have software installed to run and execute an [iPython Notebook](http://ipython.org/notebook.html)

We recommend students install [Anaconda](https://www.continuum.io/downloads), a pre-packaged Python distribution that contains all of the necessary libraries and software for this project.

### Code

Template code is provided in the `Identify_Customer_Segments.ipynb` notebook file. While some code has already been implemented to get you started, you will need to implement additional functionality when requested to successfully complete the project.

### Run

In a terminal or command window, navigate to the top-level project directory `finding_donors/` (that contains this README) and run one of the following commands:

```bash
ipython Identify_Customer_Segments.ipynb
```  
or
```bash
jupyter notebook Identify_Customer_Segments.ipynb
```

This will open the iPython Notebook software and project file in your browser.

### Overview

This project uses proprietary data set (not included in this repository due to this reason) of German population data, and consumer data from a mail-order company in Germany. The objective is to use unsupervised learning techniques to identify under and over represented customer segments between the two populations.

The project involved data wrangling to prepare the data set, which was then feature engineered using Principle Component Analysis (PCA). Finally, KMeans clusterig was used to generate segments between the two datasets, which was then compared with the larger German population data to arrive at both under and over represented segments of the customer data.
