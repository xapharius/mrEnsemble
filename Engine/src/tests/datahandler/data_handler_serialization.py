'''
Created on Mar 13, 2014

'''
import unittest
from datahandler.numerical.NumericalDataHandler import NumericalDataHandler
import pickle
import os


class DataHandlerSerialization(unittest.TestCase):


    def test_numerical_handler_serialization(self):
        test_file_name = 'num_handler.pkl'
        handler = NumericalDataHandler(10, 1)
        
        output = open(test_file_name, 'wb')
        # serialize handler with highest protocol version available
        pickle.dump(handler, output, -1)
        
        output.close()
        os.remove(test_file_name)
        assert True

    def test_numerical_handler_deserialization(self):
        test_file_name = 'num_handler.pkl'
        handler = NumericalDataHandler(10, 1)
        
        # serialize
        output = open(test_file_name, 'wb')
        # use highest protocol version available
        pickle.dump(handler, output, -1)
        output.close()
        
        # deserialize
        pkl_file = open(test_file_name, 'rb')
        loaded_handler = pickle.load(pkl_file)
        
        os.remove(test_file_name)
        
        pre_processor = loaded_handler.get_pre_processor()
        data_processor = loaded_handler.get_data_processor()
        configuration = loaded_handler.get_configuration()
        
        assert pre_processor != None
        assert data_processor != None
        assert configuration != None


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()