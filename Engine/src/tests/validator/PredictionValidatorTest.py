'''
Created on Mar 12, 2014

@author: xapharius
'''
import unittest
import numpy as np
from validator.PredictionValidator import PredictionValidator
from numpy.ma.testutils import assert_equal
from datahandler.numerical.NumericalDataSet import NumericalDataSet
from algorithms.linearRegression import LinearRegression
from numpy.testing.utils import assert_almost_equal


class Test(unittest.TestCase):


    def test_compute_mseE(self):
        targets = [np.array([[-2]]), np.array([[2]]), np.array([[4]])]
        predictions = [np.array([[1]]), np.array([[-1]]), np.array([[4]])]
        val = PredictionValidator()
        mse = val.compute_mse(targets, predictions)
        # computed by hand
        assert_equal(round(mse, 3), 6, "wrong mse")

    def test_compute_r2(self):
        targets = [np.array([[0]]), np.array([[2]]), np.array([[4]]), np.array([[6]])]
        predictions = [np.array([[0]]), np.array([[1.5]]), np.array([[3]]), np.array([[4.5]])]
        val = PredictionValidator()
        r2 = val.compute_r2(targets, predictions)
        # computed per hand ssr/sst
        # I HAVE NO IDEA WHY THE EQUATIONS ARE NOT EQUAL
        assert_equal(round(r2,3), 0.675, "wrong R-Squared")

    def test_validation(self):
        csv_source = "../../../../data/wine-quality/winequality-red.csv"
        rawData = np.genfromtxt(csv_source, delimiter=';', skip_header=0)
        inputs = rawData[:,:11]
        targets = rawData[:,11:]
        num_data = inputs.shape[0]
        
        data_set_1 = NumericalDataSet(inputs[0:num_data/2], targets[0:num_data/2])
        data_set_2 = NumericalDataSet(inputs[num_data/2:], targets[num_data/2:])

        complete_data = NumericalDataSet(np.vstack((data_set_1.get_inputs(), data_set_2.get_inputs())), targets)
        
        linreg = LinearRegression(complete_data.nrInputVars)
#         predictions = linreg.predict(complete_data)
        
        validator = PredictionValidator()
        stats_1 = validator.validate(linreg, data_set_1)
        stats_2 = validator.validate(linreg, data_set_2)
        
        result = validator.aggregate([stats_1, stats_2])
        
        assert_almost_equal(result['mse'], validator.validate(linreg, complete_data)['mse'], err_msg="Validation statistics for whole data did not match result of aggregated partial statistics.")


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.test_computeMSE']
    unittest.main()