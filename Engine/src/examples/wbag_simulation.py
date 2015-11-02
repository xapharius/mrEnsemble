'''
Created on Aug 4, 2015

@author: xapharius
'''
import numpy as np
import pandas as pd
import simulation.benchmarker.dataset_loader as dloader
from datahandler.numerical2.numerical_data_handler import NumericalDataHandler
from factory.algorithm_factory import AlgorithmFactory
from sklearn.linear_model.logistic import LogisticRegression
from factory.homogenous_factory import HomogenousFactory
from ensemble.classification.weighted_bag import WBag
from simulation.mr_simulator.wbag_simulator import WBagSimulator
from simulation.sampler.bootstrap_sampler import BootstrapSampler
from validator.classification_validator import ClassificationValidator


nr_mappers = 2
subset_of_features = False


datahandler = NumericalDataHandler(random_subset_of_features = subset_of_features)
algf = AlgorithmFactory(LogisticRegression)
manager_factory = HomogenousFactory(datahandler, algf)

rawdataset = dloader._get_bank()
sample_ratio = 1./nr_mappers
sampler = BootstrapSampler(sample_size_ratio=0.95, with_replacement=False)
results_all = pd.DataFrame()
results_change = pd.DataFrame()

print "\n\nDataset={} (n={}), input_dim={}, label_dim={}"\
    .format(rawdataset.name, rawdataset.total_obs, rawdataset.input_var, rawdataset.target_var)
sampler.bind_data(rawdataset.training_inputs, rawdataset.training_targets)

# simulation - train ensemble
simulator = WBagSimulator(data_sampler=sampler, 
    factory=manager_factory, ensemble_cls=WBag)
ensemble = simulator.simulate(nr_mappers=nr_mappers)
print "Number of Features per Model:", [manager.feature_engineer.number_of_features for manager in ensemble.managers]
print "Training Obs per model", [manager.training_data_statistics["nr_obs"] for manager in ensemble.managers]
print "Ensemble Weights", ['%.2f' % weight for weight in ensemble.weights]

# train benchmark model
benchmark_model = manager_factory.get_instance()
benchmark_model.feature_engineer.random_subset_of_features_ratio = 1
benchmark_model.train(rawdataset.training_inputs, rawdataset.training_targets) 

# validate both
validator = ClassificationValidator()
benchmark_results = validator.validate(benchmark_model, rawdataset.validation_inputs, rawdataset.validation_targets)
ensemble_results = validator.validate(ensemble, rawdataset.validation_inputs, rawdataset.validation_targets)

# append to results list/dataframe
df_b = pd.DataFrame(benchmark_results, index=[rawdataset.name+"_b"])
df_e = pd.DataFrame(ensemble_results, index=[rawdataset.name+"_e"])
results_all = pd.concat([results_all, df_b, df_e])
print "\n\nScores:\n", results_all

change = (df_e.reset_index(drop=True) / df_b.reset_index(drop=True)) - 1
change.index=[rawdataset.name]
print "\n,\nChange (0.%) to benchmark model:\n", change
