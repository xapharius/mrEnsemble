"""
Created on Feb 4, 2014

@author: xapharius
"""
from algorithms.AbstractAlgorithm import AbstractAlgorithm
import numpy as np
import utils.numpyutils as nputils
from utils import logging
from algorithms.neuralnetwork.updatemethods.simple_update import SimpleUpdate


class MultilayerPerceptron(AbstractAlgorithm):
    """
    Predictive Feed Forward Neural Network Class
    """

    def __init__(self, arr_layer_sizes, iterations=1, do_classification=False,
                  update_method=SimpleUpdate(0.5), batch_update_size=1,
                  activ_func=(nputils.sigmoid_np_arr, nputils.sigmoid_deriv)):
        """
        Creates a Prediction Neural Network - weights 
        :param arr_layer_sizes: list with number of neurons each layer should
        have. index starts with input layer.
        :param iterations: optional (default is 1), number of iterations
        performed on the given data set when doing training
        :param do_classification: (optional) if set to true last layer also
        applies the activation function, otherwise not.
        :param update_method: optional (default is SimpleUpdate with learning
        rate of 0.5)Update method to be used for weight update
        :param batch_update_size: optional (default is 1), specifies the number
        of training examples to look at before applying a weight update. A size
        of 1 is usually referred to as stochastic (or incremental) gradient 
        descent whereas a value greater 1 is known as batch gradient descent.
        """
        # Sizes for each layer, 0 is input layer
        self.do_classification = do_classification
        self.arr_layer_sizes = arr_layer_sizes
        self.nrLayers = len(arr_layer_sizes)
        self.iterations = iterations
        self.batch_update_size = batch_update_size
        self.update_method = update_method
        self.activation_function = activ_func[0]
        self.deriv_activation_function = activ_func[1]

        weights_arr = []
        for layer in range(len(arr_layer_sizes) - 1):
            # weight matrix shape is first layer * second layer
            # bias term added on first dimension
            # generate random weights in the range of [-0.5, 0.5]
            weights = np.random.rand(arr_layer_sizes[layer] + 1,
                                     arr_layer_sizes[layer + 1]) - 0.5
            weights_arr.append(weights)

        self.weights_arr = weights_arr

    def train(self, data_set):
        """
        Online Training for given dataset
        :param data_set: NumericalDataSet
        """
        for it in range(self.iterations):
            # randomly select observations as many times as there are
            # observations
            logging.info("Iteration #" + str(it + 1))
            it_error = 0
            batch_error = 0
            batch_updates = [np.zeros(np.shape(w)) for w in
                              self.weights_arr]
            for o in range(data_set.get_nr_observations()):
                input_arr, target_arr = data_set.rand_observation()
                # feed-forward
                outputs = self.feedforward(input_arr)
                current_error = self.calculate_error(target_arr, outputs[-1])
                it_error += current_error
                batch_error += current_error
                # backpropagation
                deltas = self.backpropagation(outputs, target_arr)
                weight_updates = self.calculate_weight_updates(deltas,
                                                                outputs)
                # accumulate weight updates until batch size is reached, then
                # do the weight update
                for i in range(len(self.weights_arr)):
                    batch_updates[i] += weight_updates[i]
                if (o + 1) % self.batch_update_size == 0:
                    self.update_method.perform_update(self.weights_arr,
                                                      batch_updates,
                                                      batch_error)
                    # print("  Avg. error: " + str(batch_error/self.batch_update_size) + "\n")
                    batch_error = 0
                    for j in range(len(batch_updates)):
                        batch_updates[j].fill(0)

            logging.info("  Avg. error: " + str(it_error / data_set.get_nr_observations()) + "\n")

    def predict(self, data_set):
        """
        Predicts targets for given data set
        :param data_set: data Set inheriting AbstractDataSet
        :return: outputs from the feed forward on each row
        :rtype: list of numpy.ndarray (nr_obs * nr_output_neurons)
        """
        predictions = []
        # loop through dataset
        for observation, _ in data_set.gen_observations():
            # make sure it numpy array
            inputArr = np.array(observation)
            # feedforward
            pred = self.feedforward(inputArr)[-1]
            if self.do_classification:
                pred = nputils.softmax(pred, self.weights_arr[-1].size)
            predictions.append(pred)

        return predictions

    def set_params(self, parameters):
        """
        Set parameters of predefined model (shape of parameters already specified)
        :param parameters: array of np.array
        :raise exception: if given parameters don't match in shape with model
        """
        for wIndex in range(len(parameters)):
            if self.weights_arr[wIndex].shape != parameters[wIndex].shape:
                raise Exception("overwriting parameters have not the same shape"
                                + " as the model (weight Matrix) " + str(wIndex)
                                + ".\n        model: "
                                + str(self.weights_arr[wIndex].shape)
                                + "\n  overwriting: "
                                + str(parameters[wIndex].shape))
            self.weights_arr[wIndex] = parameters[wIndex]

    def feedforward(self, input_vec):
        """
        Feed inputs forward through net.
        :param input_vec: nparray of inputs. Size defined by input layer. Row vector shape = (1,x) hint: np.array([[1,2,3]])
        :return: activations for each layer shape = (1,x).
        :rtype: array of np.Arrays(1dim), for each layer one (weight layers + 1)
        :raise exception: if given input size doesn't match with input layer
        """

        if input_vec.shape != (1, self.arr_layer_sizes[0]):
            raise Exception("Invalid inputvector shape. (1," + str(
                self.arr_layer_sizes[0]) + ") expected, got " + str(
                input_vec.shape))

        outputs = [input_vec]

        # feed forward through network
        for i in range(len(self.weights_arr)):
            # input is output of last layer plus bias
            layer_in = nputils.add_one_to_vec(outputs[-1])
            # activation is weighted sum for each neuron
            layer_activation = np.dot(layer_in, self.weights_arr[i])
            # activation function is a logistic unit, except last layer
            if self.do_classification:
                layer_out = self.activation_function(layer_activation)
            else:
                layer_out = layer_activation
            outputs.append(layer_out)

        return outputs

    def backpropagation(self, outputs, targets):
        """
        Propagates errors through NN, computing the partial gradients
        :param outputs: List of np.arrays(1,x) obtained from feedforward
        :param targets: np.array of shape (1,output_layer_size) representing the desired output
        :return: list of deltas for each weight matrix
        """
        # the deltas from the delta rule
        # deltas for the output layer - no sigmoid derivative since output is
        # linear deltas will have same shape as activations = (1,x)
        deltas = [outputs[-1] - targets]
        # starting from second last layer, iterating backwards through nn UNTIL
        # second layer (input layer doesnt need deltas)
        # weights i are between layer i and i + 1
        for i in reversed(range(1, self.nrLayers - 1)):
            # multiply weights with previous computed deltas (first in list) to
            # obtain the sum over all neurons in next layer, for each neuron
            # in current layer
            sums = np.dot(self.weights_arr[i], deltas[0].transpose())
            # remove last sum since it is from bias neuron. we don't need a
            # delta for it, since it doesn't have connections to the previous
            # layer
            sums = sums[:-1, :].transpose()
            # element-wise multiply with the sigmoidal derivative for activation
            current_delta = self.deriv_activation_function(
                outputs[i]) * sums
            # PREpend delta to array
            deltas.insert(0, current_delta)
        return deltas


    def calculate_weight_updates(self, deltas, outputs):
        """
        Calculates updates for the weights based on the given deltas and activations.
        :param deltas: List of deltas calculated by backpropagation
        :param outputs: List of outputs calculated by feed forward
        :return: List of weight updates
        """
        changes = []
        # weights i are between layer i and i + 1
        for i in range(self.nrLayers - 1):
            # here the activations need the additional bias neuron -> addOneToVec
            # unfortunately both arrays have to be transposed
            changes.append(np.dot(nputils.add_one_to_vec(outputs[i]).transpose(), deltas[i]))
        return changes

    def calculate_error(self, expected, actual):
        return np.sum(0.5 * np.power(expected - actual, 2))
