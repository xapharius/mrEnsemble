import numpy as np
from algorithms.neuralnetwork.feedforward.multilayer_perceptron import MultilayerPerceptron


class BaggedPredictionNN(object):
    
    def __init__(self, nns=None, arr_weights=None):
        if nns is not None:
            self.nets = nns
        elif arr_weights is not None:
            arr_layer_sizes = [ len(layer)-1 for layer in arr_weights[0] ]
            self.nets = []
            for weights in arr_weights:
                net = MultilayerPerceptron(arr_layer_sizes)
                net.weights_arr = [ np.array(w) for w in weights ]
                self.nets.append(net)
        else:
            raise ValueError()
    
    def predict(self, data_set):
        predictions = [ net.predict(data_set) for net in self.nets ]
        final_predictions = []
        prediction_shape = np.shape(predictions[0][0])
        for pred_idx in range(len(predictions[0])):
            pred = []
            for net_idx in range(len(predictions)):
                pred.append(np.argmax(predictions[net_idx][pred_idx]))
            pred_vec = np.zeros(prediction_shape)
            pred_vec[0, np.argmax(np.bincount(pred))] = 1
            final_predictions.append(pred_vec)
        return final_predictions
