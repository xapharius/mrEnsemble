'''
Created on May 2, 2015

@author: xapharius
'''
import unittest
from  tests.datahandler.numerical import *

def get_suite():
    numerical_test_suite = unittest.TestSuite()
    numerical_test_suite.addTests(unittest.makeSuite(NumericalDatahandlerTest))
    numerical_test_suite.addTests(unittest.makeSuite(NumericalDatasetTest))
    numerical_test_suite.addTests(unittest.makeSuite(NumericalFeatureSelectorTest))
    return numerical_test_suite

if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(get_suite())
