'''
Created on Mar 23, 2015

@author: xapharius
'''
import numpy as np

from simulation.mr_simulator.ensemble_simulator import EnsembleSimulator
from simulation.sampler.bootstrap_sampler import BootstrapSampler


from sklearn.linear_model import LinearRegression
from datahandler.numerical2.numerical_data_handler import NumericalDataHandler
from factory.algorithm_factory import AlgorithmFactory
from factory.homogenous_factory import HomogenousFactory
from ensemble.regression.bag import Bag

from validator.regression_validator import RegressionValidator
from validator.classification_validator import ClassificationValidator


if __name__ == '__main__':

    print("=== Ensemble Simulation Example ===")

    nr_params = 11
    nr_label_dim = 1
    data_file = '../../../data/wine-quality/winequality-red.csv'

    print(  "\n             data: " + data_file
          + "\n           params: " + str(nr_params)
          + "\n        target dim: " + str(nr_label_dim)
          + "\n"
          )

    # 0. Prepare Data Scource
    data = np.loadtxt(open(data_file, "rb"), delimiter = ";")
    training_data = data[:1000]
    validation_data = data[1000:]
    bsampler = BootstrapSampler(sample_size_ratio = 0.1)
    bsampler.bind_data(training_data)


    # 1. set data handler
    datahandler = NumericalDataHandler(random_subset_of_features = True)

    # 2. define algorithm Factory
    algf = AlgorithmFactory(LinearRegression)

    # 3 Factory
    factory = HomogenousFactory(datahandler, algf)

    # 4. run
    simulator = EnsembleSimulator(data_sampler = bsampler, factory = factory, ensemble_cls = Bag)
    ensemble = simulator.simulate(nr_mappers = 10)
    print "Ensemble's number of features per model:", [manager.feature_engineer.number_of_features for manager in ensemble.managers]

    # 5. validate result
    validator = RegressionValidator()

    model_results = validator.validate(ensemble, validation_data)
    print "Bagged Model:"
    print model_results

    #Benchmark
    benchmark_model = factory.get_instance()
    benchmark_model.feature_engineer.random_subset_of_features_ratio = 1                                 
    benchmark_model.train(training_data)

    benchmark_results = validator.validate(benchmark_model, validation_data)
    print "Benchmark Model:"
    print benchmark_results