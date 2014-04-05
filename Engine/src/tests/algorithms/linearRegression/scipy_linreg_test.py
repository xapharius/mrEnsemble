'''
Created on Apr 5, 2014

@author: linda
'''
import unittest

from algorithms.linearRegression.scipy_linreg import SciPyLinReg
from datahandler.numerical.NumericalDataSet import NumericalDataSet
import numpy as np
from numpy.testing.utils import assert_array_almost_equal
from algorithms.linearRegression.scipy_linreg_factory import SciPyLinRegFactory


class SciPyLinRegTest(unittest.TestCase):


    def test_training(self):
        inputs = np.array([[0, 0], [1, 1], [2, 2]])
        targets = np.array([[0], [1], [2]])
        data_set = NumericalDataSet(inputs, targets)

        lin_reg = SciPyLinReg(SciPyLinReg.ORDINARY)
        lin_reg.train(data_set)

        assert_array_almost_equal([0.5, 0.5], lin_reg.get_params())


    def test_aggregation(self):
        inputs = np.array([[0, 1, 2]])
        targets = np.array([[1]])
        data_set = NumericalDataSet(inputs, targets)

        factory = SciPyLinRegFactory()
        lin_reg_1 = factory.get_instance()
        lin_reg_1.train(data_set)
        lin_reg_2 = factory.get_instance()
        lin_reg_2.train(data_set)

        final_lin_reg = factory.aggregate([lin_reg_1, lin_reg_2])

        assert_array_almost_equal([[0., 0.2, 0.4]], final_lin_reg.get_params())


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.test_training']
    unittest.main()