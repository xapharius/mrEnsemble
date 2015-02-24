"""
Created on Jul 22, 2014

@author: Simon Hohberg
"""
import numpy as np
from algorithms.neuralnetwork.feedforward.multilayer_perceptron import MultilayerPerceptron, \
    SimpleUpdate
import utils.numpyutils as nputils
from layers import ConvLayer, MaxPoolLayer
from utils import logging
from algorithms.AbstractAlgorithm import AbstractAlgorithm


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
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.layers = []
        self.activ_func = activation_func[0]
        self.deriv_acitv_func = activation_func[1]
        num_prev_maps = 1
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

    def predict(self, data_set):
        """
        Predicts targets for given data set.
        @param data_set: data Set inheriting AbstractDataSet
        :return: List of predictions, i.e. output of this net for each
        observation in the data set.
        """
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

    def train(self, data_set):
        """
        Train net with given data set.
        :param data_set: Data set for training.
        """
        for it in range(self.iterations):
            # randomly select observations as many times as there are
            # observations
            logging.info("Iteration #" + str(it + 1))
            it_error = 0
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

            avg_error = it_error / data_set.get_nr_observations()
            logging.info("  Avg. error: " + str( avg_error ) + "\n")

    def set_params(self, parameters):
        pass
