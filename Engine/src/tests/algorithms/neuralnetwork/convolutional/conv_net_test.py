import unittest

import numpy as np

import utils.imageutils as imgutils
import utils.numpyutils as nputils
from algorithms.neuralnetwork.convolutional.conv_net import ConvNet
from datahandler.numerical.NumericalDataSet import NumericalDataSet
import utils.serialization as srlztn


def gen_vertical_bars(num):
    bars = []
    for _ in range(num):
        x, y = np.random.randint(low=0, high=15, size=2)
        length = np.random.randint(low=4, high=13)
        bar = np.zeros((16, 16))
        bar[y:y+length, x:x+2] = 1
        bars.append(bar)
    return bars

def gen_horizontal_bars(num):
    bars = []
    for _ in range(num):
        x, y = np.random.randint(low=0, high=15, size=2)
        length = np.random.randint(low=4, high=13)
        bar = np.zeros((16, 16))
        bar[y:y+2, x:x+length] = 1
        bars.append(bar)
    return bars

class Test(unittest.TestCase):

    def test_bars(self):
        # 16x16 images with bars that are 2 pixel thick
        train_verticals = gen_vertical_bars(50)
        train_horizontals = gen_horizontal_bars(50)
        test_verticals = gen_vertical_bars(50)
        test_horizontals = gen_horizontal_bars(50)
        inputs = np.array(train_verticals + train_horizontals)
        targets = np.array([[1, 0] for _ in train_verticals] + [[0, 1] for _ in train_horizontals])
        data_set = NumericalDataSet(inputs, targets)
        test_inputs = np.array(test_verticals + test_horizontals)
        test_targets = np.array([[1, 0] for _ in test_verticals] + [[0, 1] for _ in test_horizontals])
        test_data_set = NumericalDataSet(test_inputs, test_targets)

        # 16x16 -> C(3): 14x14 -> P(2): 7x7 -> C(3): 5x5 -> P(5): 1x1
        net_topo = [('c', 3, 6), ('p', 2), ('c', 3, 8), ('p', 5), ('mlp', 8, 8, 2)]
        net = ConvNet(iterations=50, learning_rate=0.001, topo=net_topo)
        net.train(data_set)

        preds = net.predict(test_data_set)
        conf_mat = nputils.create_confidence_matrix(preds, test_targets, 2)
        print "Error rate: " + str(100 - (np.sum(conf_mat.diagonal()) / np.sum(conf_mat[:, :]) * 100)) + "%"


    def test_mnist_digits(self):
        digits, labels = imgutils.load_mnist_digits('../../data/mnist-digits/train-images.idx3-ubyte', '../../data/mnist-digits/train-labels.idx1-ubyte', 300)
        targets = np.array([ nputils.vec_with_one(10, digit) for digit in labels ])
        train_data_set = NumericalDataSet(np.array(digits)[:150], targets[:150])
        test_data_set = NumericalDataSet(np.array(digits)[150:], targets[150:])

        # 28x28 -> C(5): 24x24 -> P(2): 12x12 -> C(5): 8x8 -> P(2): 4x4 -> C(4): 1x1
        net_topo = [('c', 5, 8), ('p', 2), ('c', 5, 16), ('p', 2), ('c', 4, 16), ('mlp', 16, 16, 10)]
        net = ConvNet(iterations=30, learning_rate=0.01, topo=net_topo, activation_func=(nputils.rectifier, nputils.rectifier_deriv))
        net.train(train_data_set)
        try:
            srlztn.save_object('../../trained/mnist_digits.cnn', net)
        except:
            print("serialization error")

        preds = net.predict(test_data_set)
        
        conf_mat = nputils.create_confidence_matrix(preds, targets[150:], 10)
        print conf_mat
        num_correct = np.sum(conf_mat.diagonal())
        num_all = np.sum(conf_mat[:, :])
        print "Error rate: " + str(100 - (num_correct / num_all * 100)) + "% (" + str(int(num_correct)) + "/" + str(int(num_all)) + ")"


    def test_face_recognition(self):
        faces = imgutils.load_images('/home/simon/trainingdata/faces/', max_num=100)
        non_faces = imgutils.load_images('/home/simon/trainingdata/nonfaces/', max_num=100)
        faces_training = faces[0:50]
        faces_testing = faces[50:]
        non_faces_training = non_faces[0:50]
        non_faces_testing = non_faces[50:]

        inputs_training = np.array(faces_training + non_faces_training)
        targets_training = np.array([ [1, 0] for _ in range(len(faces_training))] + [ [0, 1] for _ in range(len(non_faces_training))])
        data_set_training = NumericalDataSet(inputs_training, targets_training)

        inputs_testing = np.array(faces_testing + non_faces_testing)
        targets_testing = np.array([ [1, 0] for _ in range(len(faces_testing))] + [ [0, 1] for _ in range(len(non_faces_testing))])
        data_set_testing = NumericalDataSet(inputs_testing, targets_testing)

        # 24x24 -> C(5): 20x20 -> P(2): 10x10 -> C(3): 8x8 -> P(2): 4x4 -> C(3): 2x2 -> p(2): 1x1
        net_topo = [('c', 5, 8), ('p', 2), ('c', 3, 16), ('p', 2), ('c', 3, 24), ('p', 2), ('mlp', 24, 24, 2)]
        net = ConvNet(iterations=30, learning_rate=0.01, topo=net_topo)
        net.train(data_set_training)
        preds = net.predict(data_set_testing)
        conf_mat = nputils.create_confidence_matrix(preds, targets_testing, 2)
        num_correct = np.sum(conf_mat.diagonal())
        num_all = np.sum(conf_mat[:, :])
        print "Error rate: " + str(100 - (num_correct / num_all * 100)) + "% (" + str(int(num_correct)) + "/" + str(int(num_all)) + ")"

        # fig = plt.figure(1)
        # plt.set_cmap('gray')
        # num_rows = 6x-img.shape[0]
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


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()