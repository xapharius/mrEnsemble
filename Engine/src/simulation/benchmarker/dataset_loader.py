'''
Created on May 3, 2015

@author: xapharius
'''
import os
import cv2
import numpy as np
import pandas as pd
import utils.imageutils as imgutils
import utils.numpyutils as nputils
import matplotlib.pyplot as plt
from simulation.sampler.bootstrap_sampler import BootstrapSampler


_data_folder = os.getcwd().split("Engine")[0] + "data/"
_wildfire_folder = "/media/xapharius/Storage/Documents/wildfire/"

class RawDataset(object):
    def __init__(self, training_inputs, training_targets,
        validation_inputs=None, validation_targets=None,
        name="unnamed"):
        '''
        IF VALIDATION LEFT NONE, will split training 70/30
        '''
        if validation_inputs is None:
            split = int(0.7 * len(training_inputs))
            self.validation_inputs = training_inputs[split:]
            self.validation_targets = training_targets[split:]
            self.training_inputs = training_inputs[:split]
            self.training_targets = training_targets[:split]
        else:
            self.training_inputs = training_inputs
            self.training_targets = training_targets
            self.validation_inputs = validation_inputs
            self.validation_targets = validation_targets
        self.training_obs = len(self.training_inputs)
        self.validation_obs = len(self.validation_inputs)
        self.total_obs = self.training_obs + self.validation_obs
        self.input_var = self.training_inputs[0].shape
        self.target_var = self.training_targets[0].shape
        self.name = name  


def get_datasets(data_type):
    '''
    Filters for type and task to return right set of datasets
    @return generator (name, dataset)
    '''
    if data_type == "numerical":
        return _gen_numerical_datasets()
    if data_type == "image":
        return _gen_image_datasets()
    assert "Invalid type or task"

def _gen_numerical_datasets():
    loaders = [_get_bank, _get_diabetic_retinopathy, _get_letter_recognition, _get_pendigits]
    for loader in loaders:
        yield loader()

def _gen_image_datasets():
    loaders = [_get_mnist]
    for loader in loaders:
        yield loader()

def _gen_wildfire_datasets():
    loaders = [_get_wildfire_diff, _get_wildfire_div, _get_wildfire_img2]
    for loader in loaders:
        yield loader()

def _get_bank():
    '''
    Bank-Marketing dataset "bank-full.csv"
    https://archive.ics.uci.edu/ml/datasets/Bank+Marketing
    Binary Outcome
    '''
    data = pd.read_csv(_data_folder + "bank-marketing/bank-full.csv", sep=";")
    data = data.replace(["no", "yes"], [0, 1])

    job = pd.get_dummies(data.job, prefix="job").drop("job_unknown", axis=1)
    data.drop("job", axis=1, inplace=True)
    data = pd.concat([data, job], axis=1)

    data.education = data.education.replace(['unknown', 'primary', 'secondary', 'tertiary'], range(4))
    data.marital = data.marital.replace(['unknown', 'single', 'married', 'divorced'], range(4))
    data.month = data.month.replace(['jan', 'feb', 'mar', 'apr', 'may', 'jun',
                                     'jul', 'aug', 'sep', 'oct', 'nov', 'dec'], range(1,13))

    contact = pd.get_dummies(data.contact, prefix="contact").drop("contact_unknown", axis=1)
    data.drop("contact", axis=1, inplace=True)
    data = pd.concat([data, contact], axis=1)
    outcome = pd.get_dummies(data.poutcome, prefix="poutcome").drop("poutcome_unknown", axis=1)
    data.drop("poutcome", axis=1, inplace=True)
    data = pd.concat([data, outcome], axis=1)
    # put y at end
    cols = data.columns.tolist()
    cols.remove("y")
    cols += ["y"]
    data = data[cols]
    data=data.values
    return RawDataset(name="bank-marketing", training_inputs=data[:,:-1], 
                      training_targets=data[:,-1:])

def _get_breast_cancer():
    '''
    Wisconsin Breast Cancer dataset 
    https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Original%29
    Binary Outcome
    '''
    data = pd.read_csv(_data_folder + "breast-cancer/breast-cancer-wisconsin.data", sep=",")
    data = data.replace(["?"], [np.NaN])
    data.dropna(inplace=True)
    data.label = data.label.replace([2,4], [0,1])
    data.drop("ID", axis=1, inplace=True)
    data = data.values.astype(float)
    return RawDataset(name="breast-cancer", training_inputs=data[:,:-1], 
                      training_targets=data[:,-1:])

def _get_banknote_auth():
    '''
    Banknote Authentification
    https://archive.ics.uci.edu/ml/datasets/banknote+authentication#
    Binary Outcome
    '''
    data = pd.read_csv(_data_folder + "banknote-auth/banknote_authentication.data", sep=",")
    data = data.values.astype(float)
    return RawDataset(name="banknote-auth", training_inputs=data[:,:-1], 
                      training_targets=data[:,-1:])

def _get_diabetic_retinopathy():
    '''
    Diabetic Retinopathy Debrecen Data Set Data Set
    https://archive.ics.uci.edu/ml/datasets/Diabetic+Retinopathy+Debrecen+Data+Set 
    '''
    data = pd.read_csv(_data_folder + "diabetic-retinopathy/diabetic-retinopathy.data", sep=",", header=None)
    data = data.values
    return RawDataset(name="diabetic-retinopathy", training_inputs=data[:,:-1], 
                      training_targets=data[:,-1:])

def _get_letter_recognition():
    '''
    Letter Recognition Data Set 
    https://archive.ics.uci.edu/ml/datasets/Letter+Recognition
    '''
    data = pd.read_csv(_data_folder + "letter-recognition/letter-recognition.data", sep=",", header=None)
    letters = [l for l in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"]
    data = data.replace(letters, range(len(letters)))
    # put y at end
    cols = data.columns.tolist()
    cols.remove(0)
    cols += [0]
    data = data[cols]
    data = data.values.astype(float)
    return RawDataset(name="letter-recognition", training_inputs=data[:,:-1], 
                      training_targets=data[:,-1:])

def _get_pendigits():
    '''
    Pendigits
    https://archive.ics.uci.edu/ml/datasets/Pen-Based+Recognition+of+Handwritten+Digits
    '''
    data = pd.read_csv(_data_folder + "pendigits/pendigits.data", sep=",", header=None)
    data = data.values.astype(float)
    return RawDataset(name="pendigits", training_inputs=data[:,:-1], 
                      training_targets=data[:,-1:])

def _get_mnist(nr_obs=-1):
    '''
    '''
    tin, ttar = imgutils.load_mnist_digits(_data_folder + "mnist-digits/train-images.idx3-ubyte", 
        _data_folder + "mnist-digits/train-labels.idx1-ubyte", nr_obs)
    vin, vtar = imgutils.load_mnist_digits(_data_folder + "mnist-digits/t10k-images.idx3-ubyte", 
        _data_folder + "mnist-digits/t10k-labels.idx1-ubyte", nr_obs)
    return RawDataset(name="mnist", training_inputs=tin, training_targets=ttar, 
                      validation_inputs=vin, validation_targets=vtar)

def _get_binary_mnist():
    tin, ttar = imgutils.load_mnist_digits(_data_folder + "mnist-digits/train-images.idx3-ubyte", 
        _data_folder + "mnist-digits/train-labels.idx1-ubyte", 5000)
    ttar = ttar.argmax(axis=1)
    indices = (ttar == 0) | (ttar == 1)
    tin = tin[indices]
    ttar = ttar[indices]
    ttar = ttar.reshape(len(ttar), 1)
    return RawDataset(name="binary_mnist", training_inputs=tin, training_targets=ttar)

def _get_wildfire(set_name):
    '''
    '''
    files = pd.read_csv(_wildfire_folder + "multi_files.csv", header=None, names=["file", "label"])
    labels = files.label.values[:, np.newaxis]
    data = []
    for file_name in files.file:
        img = cv2.imread(_wildfire_folder + set_name + "/" + file_name + ".png", -1)
        img = cv2.resize(img, (256, 256))
        data.append(img)
    data = np.array(data)
    return RawDataset(name="wildfire_"+set_name, training_inputs=data, training_targets=labels)

def _get_wildfire_diff():
    return _get_wildfire("diff")

def _get_wildfire_div():
    return _get_wildfire("div")

def _get_wildfire_img2():
    return _get_wildfire("img2")

"""
rawdataset = _get_wildfire("diff")
bs  = BootstrapSampler(0.03, with_replacement=False)
bs.bind_data(rawdataset.training_inputs, rawdataset.training_targets)
inp, lab = bs.sample()
for i in range(10):
    plt.figure()
    plt.imshow(inp[i], cmap="gray")
print lab[:10]
plt.show()
"""

