'''
Created on May 2, 2015

@author: xapharius
'''
import unittest
import tests.ensemble.regression.regression_suite as regression

def get_suite():
    ensemble_test_suite = unittest.TestSuite()
    ensemble_test_suite.addTests(regression.get_suite())
    return ensemble_test_suite

if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(get_suite())

