'''
Created on Aug 7, 2015

@author: xapharius
'''
from datahandler.image2.image_data_handler import ImageDataHandler
from factory.algorithm_factory import AlgorithmFactory
from algorithms.neuralnetwork.convolutional.conv_net import ConvNet
from factory.homogenous_factory import HomogenousFactory
from simulation.sampler.bootstrap_sampler import BootstrapSampler
from simulation.benchmarker.model_benchmarker import ModelBenchmarker


datahandler = ImageDataHandler()
# 28x28 -> C(5): 24x24 -> P(2): 12x12 -> C(5): 8x8 -> P(2): 4x4 -> C(4): 1x1
topo = [[('c', 5, 8), ('p', 2), ('c', 5, 16), ('p', 2), ('c', 4, 16), ('mlp', 16, 16, 10)],
        [('c', 5, 16), ('p', 2), ('c', 5, 32), ('p', 2), ('c', 4, 32), ('mlp', 32, 32, 10)],
        [('c', 7, 16), ('p', 2), ('c', 4, 32), ('p', 2), ('c', 4, 32), ('mlp', 32, 32, 10)],]
params = {'iterations':25, 'learning_rate':0.01, 'topo':topo}
algf = AlgorithmFactory(ConvNet, algorithm_params=params)
factory = HomogenousFactory(datahandler, algf)


sampler = BootstrapSampler(with_replacement=False)
bm = ModelBenchmarker("image", sampler, nr_mappers=10)


results_change, results_all = bm.benchmark(factory)
print "\n\nScores:\n", results_all
print "\n,\nChange (0.%) to benchmark model:\n", results_change