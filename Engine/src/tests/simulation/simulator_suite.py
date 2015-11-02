'''
Created on Apr 25, 2015

@author: xapharius
'''


import unittest
from tests.simulation.sampler.abstract_sampler_test import AbstractSamplerTest
from tests.simulation.sampler.bootstrap_sampler_test import BootstrapSamplerTest
from tests.simulation.benchmarker.dataset_loader_test import DatasetLoaderTest

def get_suite():
    simulator_test_suite = unittest.TestSuite()
    simulator_test_suite.addTests(unittest.makeSuite(AbstractSamplerTest))
    simulator_test_suite.addTests(unittest.makeSuite(BootstrapSamplerTest))
    simulator_test_suite.addTests(unittest.makeSuite(DatasetLoaderTest))
    return simulator_test_suite

if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(get_suite())
