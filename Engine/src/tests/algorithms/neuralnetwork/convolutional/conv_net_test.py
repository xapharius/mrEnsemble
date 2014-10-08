
import unittest
import utils.imageutils as imgutils
import numpy as np
from convnet.conv_net import ConvNet
from datahandler.numerical.NumericalDataSet import NumericalDataSet

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

        # 16x16 -> C(3): 14x14 -> P(2): 7x7 -> C(3): 5x5 -> P(2): 3x3 -> C(3): 1x1
        net_topo = [('c', 3, 4), ('p', 2), ('c', 3, 8), ('p', 2), ('c', 3, 16), ('mlp', 16, 32, 2)]
        net = ConvNet(iterations=10, learning_rate=0.001, topo=net_topo)
        net.train(data_set)

        preds = net.predict(test_data_set)
        conf_mat = np.zeros((2, 2))
        for t, p in zip([np.argmax(t) for t in test_targets], [np.argmax(x) for x in preds]):
            conf_mat[t, p] += 1
        print conf_mat
        print "Error rate: " + str(np.sum(conf_mat.diagonal()) / np.sum(conf_mat[:, :]) * 100) + "%"


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()