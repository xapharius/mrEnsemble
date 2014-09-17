"""
Created on Aug 26, 2014

@author: Simon Hohberg
"""
import numpy as np
import scipy.signal.signaltools as signal
import utils.numpyutils as nputils


class ConvLayer(object):
    """
    Convolution layer in a convolutional neural network. When doing feedforward
    this layer convolves the inputs with randomly initialized kernels (filters)
    of the configured size. The number of neurons is the same as the number of
    output feature maps. There is a kernel for each previous feature map (input)
    and feature map of this layer (fully connected).
    """

    def __init__(self, num_prev_maps, num_maps, kernel_size, activation_func=np.tanh, deriv_activation_func=nputils.tanhDeriv):
        """
        Creates a new fully connected convolution layer with a kernel for each
        previous feature map and feature map of this layer resulting in
        num_prev_maps * num_maps kernels plus a bias for each kernel. Kernels
        and biases are randomly initialized in the range [-0.5, 0.5].
        :param num_prev_maps: Number of previous layers' output feature maps
        :param num_maps: Number of output feature maps for this layer, i.e.
        number of neurons
        :param kernel_size: Size of the kernels (filters) to be used for
        convolution
        :param activation_func: Activation function that is used by the neurons
        :param deriv_activation_func: Derivative function for given
        activation function. Calculates the derivative from the activation
        functions outputs.
        """
        self.num_prev_maps = num_prev_maps
        self.num_maps = num_maps
        self.kernel_size = kernel_size
        fan_in = np.sqrt(kernel_size * kernel_size * num_prev_maps)
        self.weights = np.random.uniform(low=-1/fan_in, high=1/fan_in, size=(num_maps, num_prev_maps, kernel_size, kernel_size))
        self.biases = np.zeros(num_maps) # np.random.rand(num_maps)-0.5
        self.activation_func = activation_func
        self.deriv_activation_func = deriv_activation_func
        self.inputs = None
        self.outputs = None
        self.deltas = None
        self.gradients = None

    def feedforward(self, inputs):
        """
        Calculates output of this layer from the given input.
        :param inputs: 3D or 2D numpy array, if 3D, first dimension: idx of prev
        feature map, second and third dimension: image output of this feature
        map, if 2D just a single image.
        :return 3D numpy array, 2D numpy array output for each feature map
        """
        if len(np.shape(inputs)) == 2:
            inputs = np.array([inputs])
        self.inputs = np.copy(inputs)
        in_size = np.shape(self.inputs[0])
        out_size = in_size[0] - self.kernel_size + 1
        self.outputs = np.zeros((self.num_maps, out_size, out_size))
        # go through all feature maps of this layer
        for fm_idx in range(self.num_maps):
            bias = self.biases[fm_idx]
            conv_out = np.zeros((out_size, out_size))
            # convolve inputs with weights and sum the results
            for prev_fm_idx in range(self.num_prev_maps):
                kernel = self.weights[fm_idx, prev_fm_idx]
                prev_out = self.inputs[prev_fm_idx]
                conv_out += signal.convolve2d(prev_out, kernel, mode='valid')
            # add bias and apply activation function for final output
            self.outputs[fm_idx, :, :] = self.activation_func(conv_out + bias)
        if out_size == 1:
            return np.array([ self.outputs[:, 0, 0] ])
        return self.outputs

    def backpropagate(self, error):
        """
        TODO
        :param error: Error for this layer (backpropagated error)
        :return Error of the previous layer
        """
        if self.outputs is None:
            raise ValueError("Feedforward has to be performed before backpropagating!")

        out_size = np.shape(self.outputs[0])
        # has the same size as the input for each previous feature map
        backprop_error = np.zeros((self.num_prev_maps, out_size[0] + self.kernel_size - 1, out_size[1] + self.kernel_size - 1))
        self.deltas = np.zeros((self.num_maps, out_size[0], out_size[1]))
        for fm_idx in range(self.num_maps):
            fm_error = error[fm_idx]
            self.deltas[fm_idx] = fm_error * self.deriv_activation_func(self.outputs[fm_idx])
            # calculate error for previous layer
            for prev_fm_idx in range(self.num_prev_maps):
                # convolve delta with kernel using 'full' mode, to obtain the
                # error for the feature map in the previous layer
                kernel = self.weights[fm_idx, prev_fm_idx]
                # Todo: could use correlate2d
                backprop_error[prev_fm_idx] += nputils.rot180(signal.convolve2d(nputils.rot180(self.deltas[fm_idx]), kernel, mode='full'))

        return backprop_error

    def calc_gradients(self):
        self.gradients = np.zeros((self.num_maps, self.num_prev_maps, self.kernel_size, self.kernel_size))
        for fm_idx in range(self.num_maps):
            for prev_fm_idx in range(self.num_prev_maps):
                self.gradients[fm_idx, prev_fm_idx, :, :] = signal.convolve2d(self.inputs[prev_fm_idx], self.deltas[fm_idx], mode='valid')

    def update(self, learning_rate):
        for fm_idx in range(self.num_maps):
            self.biases[fm_idx] -= learning_rate * np.sum(self.gradients[fm_idx, :, :, :])
            for prev_fm_idx in range(self.num_prev_maps):
                self.weights[fm_idx, prev_fm_idx] -= learning_rate * self.gradients[fm_idx, prev_fm_idx, :, :]

class MaxPoolLayer(object):
    """
    Layer that takes a number of feature maps and applies max-pooling producing
    the same number of feature maps as where fed into this layer when doing
    feedforward.
    """

    def __init__(self, size):
        """
        Creates a new layer that applies max pooling to each non-overlapping
        size * size square of the given inputs.
        :param size: Size of the square that is used for max pooling
        """
        self.size = size

    def feedforward(self, inputs):
        """
        Applies max-pooling to the given input image with this layer's factor.
        The image is scaled down by this factor.
        :param inputs: 3D numpy array, a number of images that will be
        max-pooled one by one
        :return 3D or 1D numpy array, the same number of images as the input but
        each scaled down by this layer's factor. Output is 1D (row-vector) iff
        the output of a feature map is only a single pixel.
        """
        in_size = np.shape(inputs)
        fm_out_shape = (in_size[1] / self.size, in_size[2] / self.size)
        result = np.zeros((in_size[0], fm_out_shape[0], fm_out_shape[1]))
        for fm_idx in range(np.shape(inputs)[0]):
            result[fm_idx, :, :] = max_pool(inputs[fm_idx], self.size)
        # when there is only a single pixel as output, return a vector
        if fm_out_shape == (1, 1):
            result = np.array([ result[:, 0, 0] ])
        return result

    def backpropagate(self, error):
        error_shape = np.shape(error)
        backprop_error = np.zeros((error_shape[0], error_shape[1]*self.size, error_shape[2]*self.size))
        for fm_idx in range(error_shape[0]):
            backprop_error[fm_idx, :, :] = tile(error[fm_idx], self.size)
        return backprop_error


def max_pool(img, size):
    """
    Applies max-pooling to the given 2D numpy array using non-overlapping
    squares of size * size pixels. Resulting in a 2D numpy array that is
    scaled by 1/size.
    :param img: Input 2D numpy array that is max-pooled
    :param size: Size of the square used for max-pooling
    :return: Max-pooled 2D numpy array scaled by 1/size
    """
    img_shape = np.shape(img)
    result = np.zeros((img_shape[0] / size, img_shape[1] / size))
    for row in range(0, img_shape[0]-size+1, size):
        for col in range(0, img_shape[1]-size+1, size):
            result[row/size, col/size] = np.max(img[row:row + size, col:col + size])
    return result


def tile(img, size):
    """
    Tiles each pixel in the given image 'size' times. This is meant to be used
    as inverse operation to max-pooling.
    :param img: 2D numpy array
    :param size: number how often each pixel is tiled in each dimension
    :return: 2D numpy array whose size is increased 'size' times in each
    dimension
    """
    img_shape = np.shape(img)
    result = np.zeros((img_shape[0] * size, img_shape[1] * size))
    for row in range(img_shape[0]):
        for col in range(img_shape[1]):
            result[row * size:(row + 1) * size, col * size:(col + 1) * size] = img[row, col]
    return  result