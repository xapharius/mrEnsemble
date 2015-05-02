'''
Created on May 2, 2015

@author: xapharius
'''
import unittest
from tests.factory.homogenous_factory_test import HomogenousFactoryTest

def get_suite():
    factory_test_suite = unittest.TestSuite()
    factory_test_suite.addTests(unittest.makeSuite(HomogenousFactoryTest))
    return factory_test_suite

if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(get_suite())
