"""
Created on Aug 26, 2014

@author: Simon Hohberg
"""
import numpy as np
import scipy.signal.signaltools as signal
import utils.numpyutils as nputils
import skimage.transform as trans


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
        # kernels/filters for each connection from the previous layer's feature
        # maps to the feature maps of this layer, indexes:
        # [index of feature map in previous layer,
        # index of feature map in this layer, row, column]
        # -> weights[0, 1] is the filter between feature map 0 of the previous
        # layer and feature map 1 of this layer
        self.weights = np.random.uniform(low=-1/fan_in, high=1/fan_in, size=(num_prev_maps, num_maps, kernel_size, kernel_size))
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
        out_shape = (in_size[0] - self.kernel_size + 1, in_size[1] - self.kernel_size + 1)
        self.outputs = np.zeros((self.num_maps, out_shape[0], out_shape[1]))
        # go through all feature maps of this layer
        for fm_idx in range(self.num_maps):
            bias = self.biases[fm_idx]
            conv_out = np.zeros(out_shape)
            # convolve inputs with weights and sum the results
            for prev_fm_idx in range(self.num_prev_maps):
                kernel = self.weights[prev_fm_idx, fm_idx]
                prev_out = self.inputs[prev_fm_idx]
                conv_out += signal.convolve2d(prev_out, kernel, mode='valid')
            # add bias and apply activation function for final output
            self.outputs[fm_idx] = self.activation_func(conv_out + bias)
        if out_shape == (1, 1):
            return np.array([self.outputs[:, 0, 0]])
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

        # calculate deltas for this layer
        for fm_idx in range(self.num_maps):
            fm_error = error[fm_idx]
            # calculate deltas for feature map
            # supposing that the derivation function takes the function value as
            # input
            derived_input = self.deriv_activation_func(self.outputs[fm_idx])
            self.deltas[fm_idx] = fm_error * derived_input

        # calculate errors for previous layer's feature maps: cross-correlate
        # each feature map's delta with the connection's kernel, the sum over
        # all these correlations (actually only those that have a connection to
        # the previous feature map, here: fully connected) is the delta for the
        # feature map in the previous layer
        for prev_fm_idx in range(self.num_prev_maps):
            for fm_idx in range(self.num_maps):
                # correlate delta with kernel using 'full' mode, to obtain the
                # error for the feature map in the previous layer
                kernel = self.weights[prev_fm_idx, fm_idx]
                # 'full' mode pads the input on all sides with zeros increasing
                # the overall size of the input by kernel_size-1 in both
                # dimensions ( (kernel_size-1)/2 on each side)
                fm_error = nputils.rot180(signal.correlate2d(kernel, self.deltas[fm_idx], mode='full', boundary='wrap'))
                backprop_error[prev_fm_idx] += fm_error

        return backprop_error

    def calc_gradients(self):
        self.gradients = np.zeros((self.num_prev_maps, self.num_maps, self.kernel_size, self.kernel_size))
        for fm_idx in range(self.num_maps):
            for prev_fm_idx in range(self.num_prev_maps):
                prev_fm_output = self.inputs[prev_fm_idx]
                fm_delta = self.deltas[fm_idx]
                kernel = self.weights[prev_fm_idx, fm_idx]
                fm_gradient = nputils.rot180(signal.correlate2d(prev_fm_output, fm_delta, mode='full', boundary='wrap')[:kernel.shape[0], :kernel.shape[1]])
                self.gradients[prev_fm_idx, fm_idx] = fm_gradient

    def update(self, learning_rate):
        for fm_idx in range(self.num_maps):
            self.biases[fm_idx] -= learning_rate * np.sum(self.deltas[fm_idx])# * np.power(self.kernel_size, 2) * self.num_prev_maps
            for prev_fm_idx in range(self.num_prev_maps):
                fm_gradient = self.gradients[prev_fm_idx, fm_idx]
                self.weights[prev_fm_idx, fm_idx] -= learning_rate * fm_gradient

class MaxPoolLayer(object):
    """
    Layer that takes a number of feature maps and applies max-pooling producing
    the same number of feature maps as where fed into this layer when doing
    feedforward.
    """

    def __init__(self, size, num_maps, activation_func=np.tanh, deriv_activation_func=nputils.tanhDeriv):
        """
        Creates a new layer that applies max pooling to each non-overlapping
        size * size square of the given inputs.
        :param size: Size of the square that is used for max pooling
        """
        self.size = size
        self.num_maps = num_maps
        self.in_shape = None
        self.weights = np.random.random(num_maps) - 0.5
        self.biases = np.zeros(num_maps)
        self.activation_func = activation_func
        self.deriv_activation_func = deriv_activation_func
        self.output = None
        self.down_in = None
        self.deltas = None
        self.gradients = None

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
        self.in_shape = np.shape(inputs)
        fm_out_shape = (np.ceil(self.in_shape[1] / float(self.size)), np.ceil(self.in_shape[2] / float(self.size)))
        self.down_in = np.zeros((self.num_maps, fm_out_shape[0], fm_out_shape[1]))
        self.output = np.zeros((self.in_shape[0], fm_out_shape[0], fm_out_shape[1]))
        for fm_idx in range(np.shape(inputs)[0]):
            weight = self.weights[fm_idx]
            bias = self.biases[fm_idx]
            self.down_in[fm_idx] = max_pool(inputs[fm_idx], self.size)
            # self.output[fm_idx] = self.activation_func(weight * self.down_in[fm_idx] + bias)
            self.output[fm_idx] = self.down_in[fm_idx]
        out = self.output
        # when there is only a single pixel as output, return a vector
        if fm_out_shape == (1, 1):
            out = np.array([out[:, 0, 0]])
        return out

    def backpropagate(self, error):
        self.deltas = np.zeros(self.output.shape)
        error_shape = np.shape(error)
        backprop_error = np.zeros((error_shape[0], self.in_shape[1], self.in_shape[2]))
        for fm_idx in range(self.num_maps):
            fm_error = error[fm_idx]
            # fm_weight = self.weights[fm_idx]
            # deriv_input = self.deriv_activation_func(self.output[fm_idx])
            # self.deltas[fm_idx] = deriv_input * fm_error
            backprop_error[fm_idx] = tile(fm_error, self.size)#trans.resize(fm_weight * self.deltas[fm_idx], (self.in_shape[1], self.in_shape[2]))
        return backprop_error

    def calc_gradients(self):
        self.gradients = np.zeros(self.weights.shape)
        for fm_idx in range(self.num_maps):
            fm_delta = self.deltas[fm_idx]
            fm_gradient = np.sum(self.down_in[fm_idx] * fm_delta)
            self.gradients[fm_idx] = fm_gradient

    def update(self, learning_rate):
        return
        for fm_idx in range(self.num_maps):
            self.biases[fm_idx] -= learning_rate * np.sum(self.deltas[fm_idx]) #* np.power(self.kernel_size, 2) * self.num_prev_maps
            fm_gradient = self.gradients[fm_idx]
            self.weights[fm_idx] -= learning_rate * fm_gradient


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
    # pad vertically with -1
    if img_shape[0] % size != 0:
        img = np.vstack((img, np.ones((size - img_shape[0] % size, img_shape[1])) * -1))
        img_shape = np.shape(img)
    # pad horizontally with -1
    if img_shape[1] % size != 0:
        img = np.hstack((img, np.ones((img_shape[0], size - img_shape[1] % size)) * -1))
        img_shape = np.shape(img)
    result = np.zeros((img_shape[0] / size, img_shape[1] / size))
    for row in range(0, img_shape[0]-size+1, size):
        for col in range(0, img_shape[1]-size+1, size):
            result[row/size, col/size] = np.max(img[row:row + size, col:col + size])
    return result

def avg_pool(img, size):
    img_shape = np.shape(img)
    # pad vertically with -1
    # if img_shape[0] % size != 0:
    #     img = np.vstack((img, np.ones((size - img_shape[0] % size, img_shape[1])) * -1))
    #     img_shape = np.shape(img)
    # # pad horizontally with -1
    # if img_shape[1] % size != 0:
    #     img = np.hstack((img, np.ones((img_shape[0], size - img_shape[1] % size)) * -1))
    #     img_shape = np.shape(img)
    result = np.zeros((img_shape[0] / size, img_shape[1] / size))
    for row in range(0, img_shape[0]-size+1, size):
        for col in range(0, img_shape[1]-size+1, size):
            result[row/size, col/size] = np.average(img[row:row + size, col:col + size])
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