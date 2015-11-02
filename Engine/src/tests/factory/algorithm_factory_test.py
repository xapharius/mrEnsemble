'''
Created on Jul 28, 2015

@author: xapharius
'''
import unittest
from sklearn.linear_model import LinearRegression
from sklearn.linear_model.logistic import LogisticRegression
from factory.algorithm_factory import AlgorithmFactory

class AlgorithmFactoryTest(unittest.TestCase):

    def test_get_instance(self):
        algf = AlgorithmFactory(LinearRegression)
        inst = algf.get_instance()
        assert type(inst) == LinearRegression
        
    def test_random_params(self):
        params = {'a':2, 'b':range(1000000), 'c':[9], 'd':"abc"}
        algf = AlgorithmFactory(None, algorithm_params=params)
        p1 =  algf.get_random_params()
        p2 =  algf.get_random_params()
        assert isinstance(p1['a'], int) and isinstance(p1['b'], int) and isinstance(p1['c'], int)
        assert p1['d'] == "abc"
        assert p1['a'] == p2['a']
        assert p1['b'] != p2['b']
        assert p1['c'] == p2['c']
        assert p1['d'] == p2['d']
        
    def test_random_instance(self):
        params = {"penalty":["l2", "l1"], "C":[0.01, 0.1, 1., 10., 100.]}
        algf = AlgorithmFactory(LogisticRegression, algorithm_params=params)
        inst1 = algf.get_instance()
        #print inst1.get_params()

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()