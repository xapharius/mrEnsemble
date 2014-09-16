"""
Created on Jul 22, 2014

@author: Simon Hohberg
"""
import numpy as np
import scipy.ndimage as nd
import os
import skimage.io as io
from skimage import img_as_float
import matplotlib.pyplot as plt
from algorithms.neuralnetwork.feedforward.PredictionNN import PredictionNN, \
    SimpleUpdate
import scipy.signal.signaltools as signal
from datahandler.numerical.NumericalDataSet import NumericalDataSet
import utils.numpyutils as nputils
from layers import ConvLayer, MaxPoolLayer
from utils import logging


class ConvNet2(object):
    """
    classdocs
    """


    def __init__(self, num_feature_maps, kernel_sizes, pooling_patch_sizes, mlp_nn):
        """
        @param num_feature_maps: number of feature maps for each layer as list
        @param kernel_sizes: kernel size for each convolution layer as list
        @param pooling_patch_sizes: patch size for each pooling layer as list
        """
        self.conv_layers = []
        self.subsample_layers = []
        self.pooling_patch_sizes = pooling_patch_sizes
        self.mlp_nn = mlp_nn
        for layer in range(len(num_feature_maps)):
            num_maps = num_feature_maps[layer]
            kernel_size = kernel_sizes[layer]
            # [ [ (kernel, bias), (kernel, bias), ... ], ... ]
            # with weights -0.5 <= w <= 0.5
            self.conv_layers.append([ ( np.random.rand(kernel_size, kernel_size)-0.5, np.random.rand()-0.5 ) for _ in range(num_maps) ])
            # weight matrix for fully connected subsample units
            # num_maps weights for each unit + bias
            self.subsample_layers.append([ ( np.random.rand(), np.random.rand() ) for _ in range(num_maps) ])

    def feedforward(self, img):
        # output of layer n is at index n+1
        # input for layer n is at index n
        conv_layer_out = []
        sub_layer_out = [ [img] ]
        # iterate convolutional layers
        for layer_idx in range(len(self.conv_layers)):
            conv_layer = self.conv_layers[layer_idx]
            pool_patch_size = self.pooling_patch_sizes[layer_idx]
            sub_layer = self.subsample_layers[layer_idx]
            # iterate feature maps in the convolutional layer
            curr_conv_layer_out = []
            curr_sub_layer_out = []
            for fm_idx in range(len(conv_layer)):
                # convolve ouput of last layer
                kernel, conv_bias = conv_layer[fm_idx]
                # combine previous layer output (sum all feature maps)
                convolved = signal.convolve2d(sum(sub_layer_out[layer_idx]), kernel, mode='same')
                # add bias and activation function
                convolved = np.tanh(convolved + conv_bias)
                
                curr_conv_layer_out.append(convolved)
                
                # subsample/pooling
                sub_weight, sub_bias = sub_layer[fm_idx]
                # calculate activation for current subsample unit in this
                # layer, rescale using nearest interpolation
                sub_sampled = nd.zoom(convolved * sub_weight + sub_bias, 1./pool_patch_size, order=0)
                
                curr_sub_layer_out.append(sub_sampled)
            
            # convert last subsample output, so we can feed it into the first
            # layer of the MLP
            if layer_idx == len(self.conv_layers)-1:
                curr_sub_layer_out = np.array([ [ a[0,0] for a in curr_sub_layer_out ] ])
            # append convolution and subsample outputs
            conv_layer_out.append(curr_conv_layer_out)
            sub_layer_out.append(curr_sub_layer_out)
        
        mlp_out = self.mlp_nn.feedforward(sub_layer_out[-1])
        
        return conv_layer_out, sub_layer_out, mlp_out
    
    
    def backpropagation(self, conv_out, sub_out, mlp_out, targets):
        """

        :param conv_out:
        :param sub_out:
        :param mlp_out:
        :param targets:
        :return:
        """
        mlp_deltas = self.mlp_nn.backpropagation(mlp_out, targets)
        conv_layer_deltas = []
        
        first_sub_deltas = np.dot(self.mlp_nn.weightsArr[0], mlp_deltas[0].transpose())
        first_sub_deltas = first_sub_deltas[:-1, :]
        first_sub_deltas = sub_out[-1] * first_sub_deltas.transpose()
        # convert deltas to 2D arrays (one pixel 'images')
        sub_layer_deltas = [[np.array([[d]]) for d in first_sub_deltas[0]]]
        
        # iterate layers in reverse order
        for layer_idx in reversed(range(len(self.conv_layers))):
            conv_layer = self.conv_layers[layer_idx]
            sub_layer = self.subsample_layers[layer_idx]
            
            conv_layer_activ = conv_out[layer_idx]
            # original image is at index 0
            sub_layer_activ = sub_out[layer_idx-1]

            pool_patch_size = self.pooling_patch_sizes[layer_idx]

            # iterate feature maps in the convolutional layer
            curr_conv_layer_deltas = []
            curr_sub_layer_deltas = []
            for fm_idx in range(len(conv_layer)):
                # convolutional layer
                curr_conv_activ = conv_layer_activ[fm_idx]
                last_delta = sub_layer_deltas[0][fm_idx]
                last_sub_weight = sub_layer[fm_idx][0]
                upscaled_deltas = nd.zoom(last_delta, pool_patch_size, order=0)
                # TODO: only one weight?
                conv_delta = last_sub_weight * (nputils.tanhDeriv(curr_conv_activ) * upscaled_deltas)
                curr_conv_layer_deltas.append(conv_delta)
                
                kernel, conv_bias = conv_layer[fm_idx]
                
                # sub-sampling layer
                kernel = np.rot90(np.rot90(kernel))
                # TODO: derive subsample layer's activations?
                sub_delta = signal.convolve2d(conv_delta, kernel, mode='same')
                curr_sub_layer_deltas.append(sub_delta)

            conv_layer_deltas.insert(0, curr_conv_layer_deltas)
            sub_layer_deltas.insert(0, curr_sub_layer_deltas)
        
        return conv_layer_deltas, sub_layer_deltas, mlp_deltas


    def calculate_weight_update(self, conv_layer_deltas, sub_layer_deltas, mlp_deltas, conv_out, sub_out, mlp_out):
        mlp_weight_updates = self.mlp_nn.calculate_weight_updates(mlp_deltas, mlp_out)
        conv_layer_updates = []
        sub_layer_updates = []
        # iterate layers
        for layer_idx in range(len(self.conv_layers) - 1):
            conv_layer = self.conv_layers[layer_idx]
            pool_patch_size = self.pooling_patch_sizes[layer_idx]
            sub_layer = self.subsample_layers[layer_idx]
            # iterate feature maps in the layer
            curr_conv_layer_update = []
            curr_sub_layer_update = []
            for fm_idx in range(len(conv_layer)):
                conv_deltas = conv_layer_deltas[layer_idx][fm_idx]
                # convolution bias updates
                conv_bias_update = np.sum(conv_deltas)
                # kernel update
                out = sum(sub_out[layer_idx])
                conv_kernel_update = nputils.rot180(signal.convolve2d(out, nputils.rot180(conv_deltas), mode='valid'))

                if layer_idx == 0:
                    break
                # subsample update
                sub_deltas = sub_layer_deltas[layer_idx][fm_idx]
                # subsample bias update
                sub_bias_update = np.sum(sub_deltas)
                # subsample weight update
                sub_weight_update = np.sum(sub_deltas * sub_out[layer_idx+1][fm_idx])

                curr_conv_layer_update.append((conv_kernel_update, conv_bias_update))
                curr_sub_layer_update.append((sub_weight_update, sub_bias_update))

            conv_layer_updates.append(curr_conv_layer_update)
            sub_layer_updates.append(curr_sub_layer_update)

        return conv_layer_updates, sub_layer_updates, mlp_weight_updates


class ConvNet(object):

    def __init__(self, iterations=1, learning_rate=0.5, topo=[('c', 3, 4), ('p', 2), ('c', 3, 4), ('p', 9), ('mlp', 4, 4, 2)]):
        """

        :param iterations:
        :param topo
        :return:
        """
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.layers = []
        self.conv_layers = []
        num_prev_maps = 1
        # parse topology
        for layer in topo:
            # convolutional layer
            if layer[0] == 'c':
                conv_layer = ConvLayer(num_prev_maps=num_prev_maps, kernel_size=layer[1], num_maps=layer[2])
                self.add_layer(conv_layer)
                self.conv_layers.append(conv_layer)
                num_prev_maps = layer[2]
            # pooling layer
            elif layer[0] == 'p':
                self.add_layer(MaxPoolLayer(size=layer[1]))
            # multilayer perceptron
            elif layer[0] == 'mlp':
                self.mlp = PredictionNN(list(layer[1:]), update_method=SimpleUpdate(self.learning_rate))

    def add_layer(self, layer):
        """

        :param layer:
        :return:
        """
        self.layers.append(layer)

    def feedforward(self, inputs):
        """
        Feed input forward through net calculating the ouput of each layer.
        :param inputs: 2D numpy array (usually an image)
        :return: List of numpy arrays each representing the output of a layer
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

    def train(self, data_set):
        for it in range(self.iterations):
            # randomly select observations as many times as there are
            # observations
            logging.info("Iteration #" + str(it + 1))
            it_error = 0
            for o in range(data_set.get_nr_observations()):
                input_arr, target_arr = data_set.rand_observation()
                # feed-forward
                outputs = self.feedforward(input_arr)
                current_error = nputils.calc_squared_error(target_arr, outputs[-1])
                it_error += current_error

                # backpropagation
                mlp_outputs = outputs[-len(self.mlp.arrLayerSizes):]
                mlp_deltas = self.mlp.backpropagation(mlp_outputs, target_arr)
                # calculate backpropagated error of first mlp layer
                backprop_error = np.array([ [x] for x in np.dot(self.mlp.weightsArr[0], mlp_deltas[0].transpose()) ])
                for layer in reversed(self.layers):
                    backprop_error = layer.backpropagate(backprop_error)
                for conv_layer in self.conv_layers:
                    conv_layer.calc_gradients()
                    conv_layer.update(self.learning_rate)

            logging.info("  Avg. error: " + str(it_error / data_set.get_nr_observations()) + "\n")


def load_images(path, max_num=-1):
    images = []
    for _file in os.listdir(path):
        if _file.endswith('.png'):
            img = io.imread(path + '/' + _file, as_grey=True)
            images.append(img_as_float(img))
            if len(images) == max_num:
                break
    return images


if __name__ == '__main__':
    
    faces = load_images('/home/simon/Uni/Mustererkennung/uebung10/trainingdata/faces/', max_num=50)
    non_faces = load_images('/home/simon/Uni/Mustererkennung/uebung10/trainingdata/nonfaces/', max_num=50)
    inputs = np.array(faces + non_faces)
    targets = np.array([[1, 0] for _ in range(len(faces))] + [[0, 1] for _ in range(len(non_faces))])
    data_set = NumericalDataSet(inputs, targets)

    # 24x24 -> C(3): 22x22 -> P(2): 11x11 -> C(3): 9x9 -> P(3): 3x3 -> C(3): 1x1
    net = ConvNet(iterations=40, learning_rate=0.001, topo=[('c', 3, 4), ('p', 2), ('c', 3, 4), ('p', 3), ('c', 3, 4), ('mlp', 4, 4, 4, 2)])
    net.train(data_set)
    preds = net.predict(data_set)
    print preds

    # fig = plt.figure(1)
    # plt.set_cmap('gray')
    # num_rows = 6
    # num_cols = 4
    # fig.add_subplot(num_rows, num_cols, 1)
    # plt.imshow(faces[0])
    # for fm_idx in range(4):
    #     fig.add_subplot(num_rows, num_cols, num_cols*1 + fm_idx + 1)
    #     plt.imshow(convolved1[fm_idx, :, :])
    #     fig.add_subplot(num_rows, num_cols, num_cols*2 + fm_idx + 1)
    #     plt.imshow(pooled1[fm_idx, :, :])
    #     fig.add_subplot(num_rows, num_cols, num_cols*3 + fm_idx + 1)
    #     plt.imshow(convolved2[fm_idx, :, :])
    #     fig.add_subplot(num_rows, num_cols, num_cols*4 + fm_idx + 1)
    #     plt.imshow(np.array([[pooled2[0, fm_idx]]]), vmin=0, vmax=1)
    # fig.add_subplot(num_rows, num_cols, 21)
    # plt.imshow(np.array([[mlp_out[2][0, 0]]]), vmin=0, vmax=1)
    # fig.add_subplot(num_rows, num_cols, 22)
    # plt.imshow(np.array([[mlp_out[2][0, 1]]]), vmin=0, vmax=1)
    #
    # plt.show()
