'''
Created on Aug 5, 2015

@author: xapharius
'''
import unittest
from tests.ensemble.classification.wbag_test import WBagTest

def get_suite():
    classification_test_suite = unittest.TestSuite()
    classification_test_suite.addTests(unittest.makeSuite(WBagTest))
    return classification_test_suite

if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(get_suite())
