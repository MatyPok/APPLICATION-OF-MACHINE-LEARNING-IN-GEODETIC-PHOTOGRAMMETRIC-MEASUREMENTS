# APPLICATION OF MACHINE LEARNING IN GEODETIC PHOTOGRAMMETRIC MEASUREMENTS
Repository for bachelor thesis APPLICATION OF MACHINE LEARNING IN GEODETIC PHOTOGRAMMETRIC MEASUREMENTS by Matyas Pokorny


This repository contains the source code for the work Application of Machine Learning in Geodetic Photogrammetric Measurements.

## Overview

This work provides a review of modern methods of usage of machine and deep learning
in spatial information reconstruction with photogrammetric techniques and using these
tools for direct estimation of spatial information from single or more 2D images.
The practical part tests some of the reviewed methods by training chosen models with
supervised learning on publicly available datasets and evaluates their quality and pos-
sible application in the field of photogrammetry, geodesy and systems which rely on
transforming 2D information into 3D.

## Structure

The folder notebooks contain Jupyter notebooks (.ipynb) for training and basic predefined evaluation of the 7 models that are handled in the thesis. They were written and tested in a Google Colaboratory (Colab) (https://colab.research.google.com/) environment and don't have to run properly in others. 
Note that Colab is using Linux OS, which has a different syntax for OS commands than Mac or Windows. This will cause an immediate error when running with another OS (OS = Operating System).
When run, the scripts will at first download the Pix3D dataset from a URL, when run locally this dataset will be saved in the same folder the notebook is in, if run in Colab (recommended), the dataset will be saved in the session's memory, and no files should be saved on the device. Colab also has all the libraries and modules installed that are needed and the interactive environment secures the correct display of the functions messages and figures. 
However if run locally, ensure that you have these libraries, with at least these versions installed:

Numpy 1.25.2 

Matplotlib 3.7.1 

Tensorflow 2.15.0 

Sklearn 1.2.2 

In the notebooks, there is also an option to load and use already trained model. These models are available on request. 

The folder metacentrum contains the files for training the models in the first folder as well. Two additional files are here for communication with the metacentrum and for running batch computations. These have a lot of parameters that need to be specified. They are described in the file. For running a job on metacentrum an account is necessary https://metavo.metacentrum.cz/.
