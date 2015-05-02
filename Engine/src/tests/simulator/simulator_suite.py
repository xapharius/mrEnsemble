'''
Created on Apr 25, 2015

@author: xapharius
'''


import unittest
from tests.simulator.sampler.abstract_sampler_test import AbstractSamplerTest
from tests.simulator.sampler.bootstrap_sampler_test import BootstrapSamplerTest

def get_suite():
    simulator_test_suite = unittest.TestSuite()
    simulator_test_suite.addTests(unittest.makeSuite(AbstractSamplerTest))
    simulator_test_suite.addTests(unittest.makeSuite(BootstrapSamplerTest))
    return simulator_test_suite

if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(get_suite())
