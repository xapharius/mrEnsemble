from engine.engine import Engine
from algorithms.neuralnetwork.feedforward.PredictionNNFactory import PredictionNNFactory
from datahandler.numerical.NumericalDataHandler import NumericalDataHandler
import numpy as np


if __name__ == '__main__':
    
    
    print("=== Neural Network Prediction Example ===")
    
    nr_params = 11
    nr_label_dim = 1
    arr_layer_sizes = [ nr_params, 5, nr_label_dim ]
    data_file = 'hdfs:///user/linda/ml/data/winequality-red.csv'
    
    print(  "\n  data     : " + data_file
          + "\n  params   : " + str(nr_params)
          + "\n  label dim: " + str(nr_label_dim)
          + "\n  layers   : " + str(arr_layer_sizes)
          + "\n"
          )
    
    # 1. define algorithm
    pred_nn = PredictionNNFactory(arr_layer_sizes)
    
    # 2. set data handler (pre-processing, normalization, data set creation)
    data_handler = NumericalDataHandler(nr_params, nr_label_dim)
    
    # 3. run
    engine = Engine(pred_nn, data_handler, data_file)
    trained_alg = engine.start()
    print trained_alg.weightsArr
    
    data_processor = data_handler.get_data_processor()
    data_processor.set_data(np.array([[7.4, 0.7, 0 , 1.9, 0.076, 11, 34, 0.9978, 3.51, 0.56, 9.4, 5]]))
    data_processor.normalize_data(data_handler.get_statistics())
    test_set = data_processor.get_data_set()
    pred = trained_alg.predict(test_set)
    
    print(pred)
    print(test_set.targets)
    
    # 4. do something good
