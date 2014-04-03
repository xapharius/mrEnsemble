from algorithms.linearRegression.LinearRegressionFactory import LinearRegressionFactory
from datahandler.numerical.NumericalDataHandler import NumericalDataHandler
from engine.engine import Engine
from validator.PredictionValidator import PredictionValidator
from datahandler.numerical.NumericalDataProcessor import NumericalDataProcessor
from engine.constants.run_type import *


if __name__ == '__main__':
    
    print("=== Linear Regression Example ===")
    
    nr_params = 11
    nr_label_dim = 1
    run_type = LOCAL
    data_file = 'hdfs:///user/linda/ml/data/winequality-red.csv' if run_type == HADOOP else '../data/wine-quality/winequality-red.csv'
    input_scalling = NumericalDataProcessor.STANDARDIZE
    target_scalling = NumericalDataProcessor.STANDARDIZE
    
    print(  "\n             data: " + data_file
          + "\n           params: " + str(nr_params)
          + "\n        label dim: " + str(nr_label_dim)
          + "\n         run type: " + run_type
          + "\n   input scalling: " + input_scalling
          + "\n  target scalling: " + target_scalling
          + "\n"
          )
    
    # 1. define algorithm
    regression = LinearRegressionFactory(nr_params)
    
    # 2. set data handler (pre-processing, normalization, data set creation)
    data_handler = NumericalDataHandler(nr_params, nr_label_dim, input_scalling=input_scalling, target_scalling=target_scalling)
    
    # 3. run
    engine = Engine(regression, data_file, data_handler=data_handler, run_type=run_type)
    trained_alg = engine.start()
    
    # 4. validate result
    validation_stats = engine.validate(trained_alg, PredictionValidator())
    print validation_stats
