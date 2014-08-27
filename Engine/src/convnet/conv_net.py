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
from algorithms.neuralnetwork.feedforward.PredictionNN import PredictionNN
import scipy.signal.signaltools as signal
import utils.numpyutils as nputils
from layers import ConvLayer, MaxPoolLayer


class ConvNet(object):
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



def load_images(path):
    images = []
    for _file in os.listdir(path):
        if _file.endswith('.png'):
            img = io.imread(path + '/' + _file, as_grey=True)
            images.append(img_as_float(img))
    return images


if __name__ == '__main__':
    
    faces = load_images('/home/simon/Uni/Mustererkennung/uebung10/trainingdata/faces/')

    # input size is 24x24

    # out size is 22x22
    conv_layer1 = ConvLayer(num_prev_maps=1, num_maps=4, kernel_size=3)
    # out size is 11x11
    pool_layer1 = MaxPoolLayer(size=2)
    # out size is 9x9
    conv_layer2 = ConvLayer(num_prev_maps=conv_layer1.num_maps, num_maps=4, kernel_size=3)
    # out size is 1x1
    pool_layer2 = MaxPoolLayer(size=9)

    mlp = PredictionNN([ 4, 4, 2 ])

    convolved1 = conv_layer1.feedforward(faces[0])
    pooled1 = pool_layer1.feedforward(convolved1)

    convolved2 = conv_layer2.feedforward(pooled1)
    pooled2 = pool_layer2.feedforward(convolved2)

    mlp_out = mlp.feedforward(pooled2)


    # backpropagation
    mlp_deltas = mlp.backpropagation(mlp_out, np.array([[1, 0]]))
    error = np.array([ [x] for x in np.dot(mlp.weightsArr[0], mlp_deltas[0].transpose()) ])
    in_error = conv_layer1.backpropagate(pool_layer1.backpropagate(conv_layer2.backpropagate(pool_layer2.backpropagate(error))))

    conv_layer1.calc_gradients()
    conv_layer2.calc_gradients()

    # net = ConvNet([ 4, 6, 8 ], [ 3, 3, 3 ], [ 2, 2, 6 ], PredictionNN([ 8, 8, 2 ]))
    #
    # conv_activ, sub_activ, mlp_activ = net.feedforward(faces[0])
    # conv_layer_deltas, sub_layer_deltas, mlp_deltas = net.backpropagation(conv_activ, sub_activ, mlp_activ, np.array([[1,0]]))
    # conv_layer_updates, sub_layer_updates, mlp_updates = net.calculate_weight_update(conv_layer_deltas, sub_layer_deltas, mlp_deltas, conv_activ, sub_activ, mlp_activ)
    
    fig = plt.figure(1)
    plt.set_cmap('gray')
    num_rows = 6
    num_cols = 4
    fig.add_subplot(num_rows, num_cols, 1)
    plt.imshow(faces[0])
    for fm_idx in range(4):
        fig.add_subplot(num_rows, num_cols, num_cols*1 + fm_idx + 1)
        plt.imshow(convolved1[fm_idx, :, :])
        fig.add_subplot(num_rows, num_cols, num_cols*2 + fm_idx + 1)
        plt.imshow(pooled1[fm_idx, :, :])
        fig.add_subplot(num_rows, num_cols, num_cols*3 + fm_idx + 1)
        plt.imshow(convolved2[fm_idx, :, :])
        fig.add_subplot(num_rows, num_cols, num_cols*4 + fm_idx + 1)
        plt.imshow(np.array([[pooled2[0, fm_idx]]]), vmin=0, vmax=1)
    fig.add_subplot(num_rows, num_cols, 21)
    plt.imshow(np.array([[mlp_out[2][0, 0]]]), vmin=0, vmax=1)
    fig.add_subplot(num_rows, num_cols, 22)
    plt.imshow(np.array([[mlp_out[2][0, 1]]]), vmin=0, vmax=1)

    plt.show()
