from algorithms.linearRegression.LinearRegressionFactory import LinearRegressionFactory
from datahandler.numerical.NumericalDataHandler import NumericalDataHandler
from engine.engine import Engine
from validator.PredictionValidator import PredictionValidator


if __name__ == '__main__':
    
    print("=== Linear Regression Example ===")
    
    nr_params = 11
    nr_label_dim = 1
    data_file = 'hdfs:///user/linda/ml/data/winequality-red.csv'
    
    print(  "\n  data     : " + data_file
          + "\n  params   : " + str(nr_params)
          + "\n  label dim: " + str(nr_label_dim)
          + "\n"
          )
    
    # 1. define algorithm
    regression = LinearRegressionFactory(nr_params)
    
    # 2. set data handler (pre-processing, normalization, data set creation)
    data_handler = NumericalDataHandler(nr_params, nr_label_dim)
    
    # 3. run
    engine = Engine(regression, 'hdfs:///user/linda/ml/data/winequality-red.csv', data_handler=data_handler)
    trained_alg = engine.start()
    
    # 4. validate result
    validation_stats = engine.validate(trained_alg, PredictionValidator())
    print validation_stats
