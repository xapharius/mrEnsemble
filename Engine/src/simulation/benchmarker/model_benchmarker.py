'''
Created on Apr 21, 2015

@author: xapharius
'''

import numpy as np
import pandas as pd

from datahandler.numerical2.numerical_data_handler import NumericalDataHandler
from sklearn.linear_model.logistic import LogisticRegression
from factory.algorithm_factory import AlgorithmFactory
from factory.homogenous_factory import HomogenousFactory

from simulation.sampler.bootstrap_sampler import BootstrapSampler
import simulation.benchmarker.dataset_loader as loader
from simulation.mr_simulator.ensemble_simulator import EnsembleSimulator

from ensemble.classification.weighted_bag import WBag
from validator.classification_validator import ClassificationValidator
from simulation.mr_simulator.wbag_simulator import WBagSimulator





class ModelBenchmarker(object):
    '''
    Class for running a comparison 
    
    !!! Two meta-parameters for experiment: nr_mappers of simulation and sample_size_ratio for sampler 
    
    '''


    def __init__(self, data_type="numerical", sampler=BootstrapSampler(), 
                 simulator=WBagSimulator, nr_mappers=10):
        '''
        Constructor - Defining an experiment/environment setting 
        in order to then benchmark different models
        @param task: "classification" or "regression"
        (to know what validation metrics to choose)
        @param sampler: unbound sampler
        @param nr_mappers: number of mappers simulator should use
        @param train_ratio: ratio of training set to total amount of data, 
        the rest will be used for validaion
        '''
        self.data_type = data_type
        self.nr_mappers = nr_mappers
        self.sampler = sampler
        self.sampler.sample_size_ratio = 1. / nr_mappers
        # iterable of RawDataset
        self.datasets = loader.get_datasets(data_type=self.data_type)
        self.simulator = simulator

    def benchmark(self, manager_factory):
        results_all = pd.DataFrame()
        results_change = pd.DataFrame()
        for rawdataset in self.datasets:
            print "\n\nDataset={} (n={}), input_dim={}, label_dim={}"\
                .format(rawdataset.name, rawdataset.total_obs, rawdataset.input_var, rawdataset.target_var)
            self.sampler.bind_data(rawdataset.training_inputs, rawdataset.training_targets)

            # simulation - train ensemble
            simulator = self.simulator(data_sampler=self.sampler, 
                factory=manager_factory, ensemble_cls=WBag)
            ensemble = simulator.simulate(nr_mappers=self.nr_mappers)
            print "Number of Features per Model:", [manager.feature_engineer.number_of_features for manager in ensemble.managers]
            print "Training Obs per model", [manager.training_data_statistics["nr_obs"] for manager in ensemble.managers]
            print "Ensemble Weights", ['%.4f' % weight for weight in ensemble.weights]
            print "Params per Model:"
            for manager in ensemble.managers:
                print manager.model.get_params()

            # train benchmark model
            benchmark_model = manager_factory.get_instance()
            benchmark_model.feature_engineer.random_subset_of_features_ratio = 1
            benchmark_model.train(rawdataset.training_inputs, rawdataset.training_targets) 

            # validate both
            validator = ClassificationValidator()
            ensemble_results = validator.validate(ensemble, rawdataset.validation_inputs, rawdataset.validation_targets)
            benchmark_results = validator.validate(benchmark_model, rawdataset.validation_inputs, rawdataset.validation_targets)

            # append to results list/dataframe
            df_b = pd.DataFrame(benchmark_results, index=[rawdataset.name+"_b"])
            df_e = pd.DataFrame(ensemble_results, index=[rawdataset.name+"_e"])
            results_all = pd.concat([results_all, df_b, df_e])

            change = (df_e.reset_index(drop=True) / df_b.reset_index(drop=True)) - 1
            change.index=[rawdataset.name]
            results_change = pd.concat([results_change, change])
        return results_change, results_all


    