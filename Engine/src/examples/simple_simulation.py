'''
Created on Mar 3, 2015

@author: xapharius
'''
from algorithms.linearRegression.scipy_linreg import SciPyLinReg
from algorithms.linearRegression.scipy_linreg_factory import SciPyLinRegFactory
from datahandler.numerical.NumericalDataHandler import NumericalDataHandler
from simulation.mr_simulator.simulator import Simulator
from simulation.sampler.bootstrap_sampler import BootstrapSampler
import numpy as np
from validator.regression_validator import RegressionValidator
from validator.classification_validator import ClassificationValidator


if __name__ == '__main__':

    print("=== Simple Simulation Example ===")

    nr_params = 11
    nr_label_dim = 1
    data_file = '../../../data/wine-quality/winequality-red.csv'

    print(  "\n             data: " + data_file
          + "\n           params: " + str(nr_params)
          + "\n        label dim: " + str(nr_label_dim)
          + "\n"
          )

    # 0. Prepare Data Scource
    data = np.loadtxt(open(data_file, "rb"), delimiter = ";")
    training_data = data[:1000]
    validation_data = data[1000:]
    bsampler = BootstrapSampler(sample_size_ratio = 0.1)
    bsampler.bind_data(training_data)

    # 1. define algorithm
    regression = SciPyLinRegFactory(SciPyLinReg.RIDGE)

    # 2. set data handler
    data_handler = NumericalDataHandler(nr_params, nr_label_dim)

    # 3. run
    simulator = Simulator(data_sampler = bsampler, data_handler = data_handler, algorithm_factory = regression)
    trained_alg = simulator.simulate(nr_mappers = 1)

    # 4. validate result
    '''
    Wont work since scipy_linreg.predict works on DataSet (does not have manager)
    '''
    validator = ClassificationValidator()
    
    """
    data_processor = data_handler.get_data_processor()
    data_processor.set_data(validation_data)
    val_data_set = data_processor.get_data_set()
    """
    
    model_results = validator.validate(trained_alg, validation_data)
    print "Aggregated Model:"
    print model_results
    
    #Benchmark
    data_processor.set_data(training_data)
    train_data_set = data_processor.get_data_set()
    
    benchmark_model = regression.get_instance()
    benchmark_model.train(train_data_set)
    
    benchmark_results = validator.validate(benchmark_model, val_data_set)
    print "Benchmark Model:"
    print benchmark_results
    
