"""
Created on Jul 22, 2014

@author: Simon Hohberg
"""
import numpy as np
from algorithms.neuralnetwork.feedforward.multilayer_perceptron import MultilayerPerceptron, \
    SimpleUpdate
import utils.numpyutils as nputils
import copy
import time
from layers import ConvLayer, MaxPoolLayer
from utils import logging
from algorithms.AbstractAlgorithm import AbstractAlgorithm
from datahandler.numerical.NumericalDataSet import NumericalDataSet
import matplotlib.pyplot as plt

class ConvNet(AbstractAlgorithm):

    def __init__(self, iterations=1, learning_rate=0.5, topo=[('c', 3, 4), ('p', 2), ('c', 3, 4), ('p', 9), ('mlp', 4, 4, 2)], activation_func=(np.tanh, nputils.tanh_deriv)):
        """
        Creates a new convolutional neural network with the given topology
        (architecture), learning rate and number of iterations.
        :param iterations: number of iterations for training.
        :param learning_rate: rate for updating the weights
        :param topo: defines the architecture of the net. It is a list of
        tuples. Each tuple represents a layer, where the first element is a
        character that specifies the type of layer. E.g. 'c' convolutional
        layer, 'p' pooling layer, 'mlp' fully connected conventional neural
        network. The next elements in the tuple are layer
        specific.
        Convolutional: 2nd element defines the kernel size, e.g. 3 for
        a 3x3 kernel. 3rd element specifies the number of maps in the layer.
        Pooling: 2nd element defines the pool patch size, e.g. 2 for a pool
        patch size of 2x2.
        MLP: each element defines the layer size for the network.
        A complete example looks like this: [('c', 3, 4), ('p', 2), ('c', 3, 4),
        ('p', 9), ('mlp', 4, 4, 2)]
        """
        self.split_ratio = 0.8
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.layers = []
        self.activ_func = activation_func[0]
        self.deriv_acitv_func = activation_func[1]
        num_prev_maps = 1
        self.topo = topo
        # parse topology
        for layer in topo:
            # convolutional layer
            if layer[0] == 'c':
                conv_layer = ConvLayer(num_prev_maps=num_prev_maps, kernel_size=layer[1], num_maps=layer[2])
                self.add_layer(conv_layer)
                num_prev_maps = layer[2]
            # pooling layer
            elif layer[0] == 'p':
                self.add_layer(MaxPoolLayer(layer[1], num_prev_maps))
            # multilayer perceptron
            elif layer[0] == 'mlp':
                self.mlp = MultilayerPerceptron(list(layer[1:]), do_classification=True, update_method=SimpleUpdate(self.learning_rate), activ_func=(self.activ_func, self.deriv_acitv_func))

    def add_layer(self, layer):
        """
        Adds the given layer to this network.
        :param layer: layer that is added
        """
        self.layers.append(layer)

    def feedforward(self, inputs):
        """
        Feed input forward through net calculating the ouput of each layer.
        :param inputs: 3D numpy array (usually a list of images)
        :return: List of 3D numpy arrays each representing the output of a layer
        except the first array in the list which is the input.
        """
        outputs = [inputs]
        for layer in self.layers:
            outputs.append(layer.feedforward(outputs[-1]))
        outputs.extend(self.mlp.feedforward(outputs[-1])[1:])
        return outputs

    def predict(self, inputs):
        predictions = self.predict_extended(inputs)
        if predictions[0].shape == (1,1):
            #binary output
            predictions = np.array(predictions).ravel()
            predictions[predictions <= 0] = 0
            predictions[predictions > 0] = 1
            return predictions[:, np.newaxis].astype(int)
        # multiclass
        sparse = np.zeros((len(predictions), predictions[0].shape[1]))
        for ix, _ in enumerate(sparse):
            sparse[ix][predictions[ix].argmax()] = 1
        assert sparse.sum() == len(predictions)
        return sparse

    def predict_extended(self, inputs):
        """
        Predicts targets for given data set.
        @param data_set: data Set inheriting AbstractDataSet
        :return: List of predictions, i.e. output of this net for each
        observation in the data set.
        """
        data_set = NumericalDataSet(inputs)
        predictions = []
        # loop through dataset
        for observation, _ in data_set.gen_observations( ):
            # make sure it is a numpy array
            input_arr = np.array(observation)
            outputs = self.feedforward(input_arr)
            predictions.append(outputs[-1])
        return predictions

    def predict_single(self, input_arr):
        """
        Predict class for a single observation.
        :param input_arr: Observation
        :return: Prediction for given observation
        """
        return self.feedforward(input_arr)[-1]

    def fit(self, inputs, targets):
        """
        Train net with given data set.
        :param data_set: Data set for training.
        n times random sampling for online learning 
        """
        split_point = int(len(inputs) * self.split_ratio)
        data_set = NumericalDataSet(inputs[:split_point], targets[:split_point])
        val_in = inputs[split_point:]
        val_targets = targets[split_point:]
        prev_layers = None
        prev_mlp = None

        self.train_acc_err = []
        self.val_acc_err = []

        for it in range(self.iterations):
            # randomly select observations as many times as there are
            # observations
            it_error = 0
            start = time.time()
            for _ in range(data_set.get_nr_observations()):
                input_arr, target_arr = data_set.rand_observation()
                # feed-forward
                outputs = self.feedforward(input_arr)
                current_error = nputils.calc_squared_error(target_arr, outputs[-1])
                it_error += current_error

                # mlp backpropagation and gradient descent
                mlp_outputs = outputs[-len(self.mlp.arr_layer_sizes):]
                mlp_deltas = self.mlp.backpropagation(mlp_outputs, target_arr)
                mlp_weight_updates = self.mlp.calculate_weight_updates(mlp_deltas, mlp_outputs)
                self.mlp.update_method.perform_update(self.mlp.weights_arr, mlp_weight_updates, current_error)
                # layer backpropagation and gradient descent
                # calculate backpropagated error of first mlp layer
                backprop_error = np.array([[x] for x in np.dot(self.mlp.weights_arr[0], mlp_deltas[0].transpose())])
                for layer in reversed(self.layers):
                    backprop_error = layer.backpropagate(backprop_error)
                # calculate the weight gradients and update the weights
                for layer in self.layers:
                    layer.calc_gradients()
                    layer.update(self.learning_rate)

            avg_error = it_error / data_set.nrObservations
            acc_err = self._accuracy_err(inputs, targets)
            self.train_acc_err.append(acc_err)

            #validation error
            acc_err = self._accuracy_err(val_in, val_targets)
            self.val_acc_err.append(acc_err)

            logging.info("Iteration #{} MSE: {}, TrainErr: {:.6f}, ValErr: {:.6f} ({:.2f}s)\n"\
                         .format(it + 1, avg_error, self.train_acc_err[-1], self.val_acc_err[-1], time.time()-start))

            #break cond
            if it > 3 and val_in is not None and self.val_acc_err[-1] > self.val_acc_err[-4]:
                # revert
                self.layers = prev_layers
                self.mlp = prev_mlp
                plt.figure()
                plt.plot(self.train_acc_err)
                plt.plot(self.val_acc_err)
                plt.show(block=False)
                break

            #prev
            if it > 0:
                prev_layers = copy.deepcopy(self.layers)
                prev_mlp = copy.deepcopy(self.mlp)


    def _accuracy_err(self, inputs, targets):
        if targets.shape[1] == 1:
            predictions = self.predict(inputs)
            acc_err = 1 - (predictions == targets).sum() / float(len(inputs))
        else:
            predictions = self.predict_extended(inputs)
            acc_err = 1 - ((np.vstack(predictions)).argmax(axis=1)==targets.argmax(axis=1)).sum() / float(len(inputs))
        return acc_err
    
    def set_params(self, parameters):
        pass
    
    def get_params(self):
        dct = {}
        dct["learning_rate"] = self.learning_rate
        dct["topo"] = self.topo
        return dct
