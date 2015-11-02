'''
Created on Aug 3, 2015

@author: xapharius
'''
from simulation.sampler.bootstrap_sampler import BootstrapSampler
from simulation.benchmarker.model_benchmarker import ModelBenchmarker
from datahandler.numerical2.numerical_data_handler import NumericalDataHandler
from factory.algorithm_factory import AlgorithmFactory
from factory.homogenous_factory import HomogenousFactory
from sklearn.linear_model.logistic import LogisticRegression
from _functools import partial

sampler = BootstrapSampler(with_replacement=False)
bm = ModelBenchmarker(sampler=sampler, nr_mappers=10)

datahandler = NumericalDataHandler(random_subset_of_features=False)
#params = {"penalty":["l2", "l1"], "C":[0.01, 0.1, 1., 10., 100.]}
params = None
algf = AlgorithmFactory(LogisticRegression, algorithm_params=params)
factory = HomogenousFactory(datahandler, algf)
results_change, results_all = bm.benchmark(factory)
print "\n\nScores:\n", results_all
print "\n,\nChange (0.%) to benchmark model:\n", results_change
