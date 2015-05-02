'''
Created on May 2, 2015

@author: xapharius
'''
import unittest
import tests.datahandler.numerical.numerical_suite as numerical_suite 

def get_suite():
    datahandler_test_suite = unittest.TestSuite()
    datahandler_test_suite.addTests(numerical_suite.get_suite())
    return datahandler_test_suite

if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(get_suite())