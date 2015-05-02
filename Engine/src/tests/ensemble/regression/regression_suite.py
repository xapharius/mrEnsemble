'''
Created on May 2, 2015

@author: xapharius
'''
import unittest
from tests.ensemble.regression.bag_test import BagTest

def get_suite():
    regression_test_suite = unittest.TestSuite()
    regression_test_suite.addTests(unittest.makeSuite(BagTest))
    return regression_test_suite

if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(get_suite())
