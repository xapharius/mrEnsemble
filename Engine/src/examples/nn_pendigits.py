from matplotlib.pyplot import plot, show

from algorithms.neuralnetwork.feedforward.PredictionNNFactory import \
    PredictionNNFactory
from datahandler.numerical.NumericalDataHandler import NumericalDataHandler
from datahandler.numerical.NumericalDataProcessor import NumericalDataProcessor
from engine.constants.run_type import HADOOP, LOCAL, INLINE, EMR
from engine.engine import Engine
import numpy as np
from validator.PredictionValidator import PredictionValidator
from datahandler.numerical.pen_digits_data_proc import PenDigitsDataProcessor
from algorithms.neuralnetwork.updatemethods.rprop import Rprop
from algorithms.neuralnetwork.feedforward.multilayer_perceptron import MultilayerPerceptron
from algorithms.neuralnetwork.feedforward.BaggedPredictionNN import BaggedPredictionNN
import utils.numpyutils as nputils


if __name__ == '__main__':
    
    
    print("=== Neural Network Prediction Example ===")
    
    nr_params = 16
    nr_label_dim = 1
    arr_layer_sizes = [ nr_params, 16, 10 ]
    iterations = 100
    batch_update_size = 10
    lines_per_map = 300
    update_method = Rprop(arr_layer_sizes, init_step=0.005)
    run_type = EMR
    data_file = '../data/pendigits-training.txt'
    validation_data_file = '../data/pendigits-testing.txt'
    input_scalling = PenDigitsDataProcessor.NORMALIZE
    
    print(  "\n             data: " + data_file
          + "\n  validation data: " + validation_data_file
          + "\n           layers: " + str(arr_layer_sizes)
          + "\n       iterations: " + str(iterations)
          + "\n       batch size: " + str(batch_update_size)
          + "\n    lines per map: " + str(lines_per_map)
          + "\n    update method: " + str(update_method.__class__)
          + "\n   input scalling: " + str(input_scalling)
          + "\n         run type: " + run_type
          + "\n"
          )
    
    # 1. define algorithm
    pred_nn = PredictionNNFactory(arr_layer_sizes, iterations=iterations, update_method=update_method, batch_update_size=batch_update_size, do_classification=True, activ_func=(nputils.rectifier, nputils.rectifier_deriv))
    
    # 2. set data handler (pre-processing, normalization, data set creation)
    data_handler = NumericalDataHandler(nr_params, nr_label_dim)
    data_handler.LINES_PER_MAP = lines_per_map
    data_handler.set_data_processor(PenDigitsDataProcessor(nr_params, input_scalling=input_scalling))
    
    # 3. run
    engine = Engine(pred_nn, data_file, data_handler=data_handler)
    trained_alg = engine.start(_run_type=run_type)
    
    # 4. validate result
    validation_stats = engine.validate(trained_alg, PredictionValidator(), _run_type=run_type, data_file=validation_data_file)
    targets = np.array(validation_stats['targets'])
    pred = np.array(validation_stats['pred'])
    
    conf_matrix = np.zeros((10,10))
    conf_matrix = np.concatenate( ( [np.arange(0, 10)], conf_matrix ), axis=0)
    conf_matrix = np.concatenate( ( np.transpose([np.arange(-1, 10)]), conf_matrix ), axis=1)
    for i in range(len(targets)):
        conf_matrix[ np.where( targets[i]==1 )[0][0]+1, np.argmax(pred[i])+1 ] += 1
    
    print("Detection rate: " + str( np.sum(np.diagonal(conf_matrix[1:,1:])) / len(targets)))
    print(str(conf_matrix))
