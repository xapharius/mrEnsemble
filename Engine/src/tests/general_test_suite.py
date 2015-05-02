'''
Created on Apr 25, 2015

@author: xapharius
'''

import unittest
import tests.manager.manager_suite as manager
import tests.simulator.simulator_suite as simulator
import tests.datahandler.datahandler_suite as data_handler
import tests.factory.factory_suite as factory
import tests.ensemble.ensemble_suite as ensemble

general_test_suite = unittest.TestSuite()
general_test_suite.addTests(manager.get_suite())
general_test_suite.addTests(simulator.get_suite())
general_test_suite.addTests(data_handler.get_suite())
general_test_suite.addTests(factory.get_suite())
general_test_suite.addTests(ensemble.get_suite())

runner = unittest.TextTestRunner(verbosity=2)
runner.run(general_test_suite)


