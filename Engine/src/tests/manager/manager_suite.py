'''
Created on Apr 25, 2015

@author: xapharius
'''
import unittest
from tests.manager.model_manager_test import ModelManagerTest

def get_suite():
    model_manager_test_suite = unittest.TestSuite()
    model_manager_test_suite.addTests(unittest.makeSuite(ModelManagerTest))
    return model_manager_test_suite

if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(get_suite())
