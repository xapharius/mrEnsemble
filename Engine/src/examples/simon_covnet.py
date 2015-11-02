'''
Created on Jul 31, 2015

@author: xapharius
'''
import numpy as np
import utils.imageutils as imgutils
import utils.numpyutils as nputils
from algorithms.neuralnetwork.convolutional.conv_net import ConvNet
from datahandler.image2.image_data_handler import ImageDataHandler
from factory.algorithm_factory import AlgorithmFactory
from datahandler.numerical.NumericalDataSet import NumericalDataSet
from factory.homogenous_factory import HomogenousFactory
import utils.serialization as srlztn
import matplotlib.pyplot as plt
from simulation.benchmarker.model_benchmarker import ModelBenchmarker
from simulation.sampler.bootstrap_sampler import BootstrapSampler
import simulation.benchmarker.dataset_loader as dloader
from validator.classification_validator import ClassificationValidator

rawdataset = dloader._get_wildfire("div")
bs  = BootstrapSampler(0.01, with_replacement=False)
bs.bind_data(rawdataset.training_inputs, rawdataset.training_targets)
inp, lab = bs.sample()
print len(lab), lab.sum()

"""
rawdataset = dloader._get_binary_mnist()
inp = rawdataset.training_inputs
lab = rawdataset.training_targets
"""

# 28x28 -> C(5): 24x24 -> P(2): 12x12 -> C(5): 8x8 -> P(2): 4x4 -> C(4): 1x1
#topo = [[('c', 5, 8), ('p', 2), ('c', 5, 16), ('p', 2), ('c', 4, 16), ('mlp', 16, 16, 1)]]
"""
# 512x -> C(101): 412x -> P(4): 103x -> C(44): 60x -> P(2) -> 30 -> C(30)
topo = [[('c', 101, 16), ('p', 4), ('c', 44, 8), ('p', 2), ('c', 30, 8), ('mlp', 8, 8, 1)]]
# 256x -> C(57): 200x -> P(4): 50x -> C(21): 30x -> P(2) -> 15 -> C(15)
topo = [[('c', 57, 16), ('p', 4), ('c', 21, 8), ('p', 2), ('c', 15, 8), ('mlp', 8, 8, 1)]]
# 128x -> C(29): 100x -> P(2): 50x -> C(11): 40x -> P(2) -> 20 -> C(20)
topo = [[('c', 29, 16), ('p', 2), ('c', 11, 8), ('p', 2), ('c', 20, 8), ('mlp', 8, 8, 1)]]
# 64x -> C(35): 30x -> P(2): 15x -> C(6): 10x -> P(2) -> 5 -> C(5)
"""
topo = [[('c', 57, 16), ('p', 10), ('c', 20, 16), ('mlp', 16, 16, 1)]]

params = {'iterations':10, 'learning_rate':0.01, 'topo':topo}
algf = AlgorithmFactory(ConvNet, algorithm_params=params)
datahandler = ImageDataHandler()
factory = HomogenousFactory(datahandler, algf)

net = factory.get_instance()
net.train(inp, lab)
#net.train(rawdataset.training_inputs, rawdataset.training_targets)

validator = ClassificationValidator()
results = validator.validate(net, rawdataset.validation_inputs, rawdataset.validation_targets)
print results

plt.plot(net.model.train_acc_err)
plt.plot(net.model.val_acc_err)
plt.show()
