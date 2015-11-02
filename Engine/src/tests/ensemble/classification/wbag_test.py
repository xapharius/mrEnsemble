'''
Created on Aug 4, 2015

@author: xapharius
'''
import numpy as np
import unittest
from ensemble.classification.weighted_bag import WBag

class WBagTest(unittest.TestCase):


    def setUp(self):
        pass


    def tearDown(self):
        pass


    def test_weighted_majority_voting_integer(self):
        wbag = WBag(list_of_managers=[0,1], list_of_weights=[0.5, 0.5])
        list_predictions = [np.array([1]), np.array([1])]
        voted = wbag._weighted_majority_voting(list_predictions)
        assert voted == 1, "same vote"

        list_predictions = [np.array([1]), np.array([0])]
        voted = wbag._weighted_majority_voting(list_predictions)
        assert voted == 0, "tie" # picks lowest

        wbag = WBag(list_of_managers=range(5), list_of_weights=[0.2, 0.1, 0.15, 0.35, 0.2])
        list_predictions = [np.array([1]), np.array([0]), np.array([0]), np.array([1]), np.array([0])]
        voted = wbag._weighted_majority_voting(list_predictions)
        assert voted == 1, "majority"

        wbag = WBag(list_of_managers=range(5), list_of_weights=[0.05, 0.05, 0.05, 0.05, 0.8])
        list_predictions = [np.array([0]), np.array([0]), np.array([0]), np.array([0]), np.array([1])]
        voted = wbag._weighted_majority_voting(list_predictions)
        assert voted == 1, "single ig weight"

    def test_weighted_majority_voting_array(self):
        wbag = WBag(list_of_managers=[0,1], list_of_weights=[0.7, 0.3])
        list_predictions = [np.array([1,0,0]), np.array([0,1,0])]
        voted = wbag._weighted_majority_voting(list_predictions)
        assert (voted == np.array([1,0,0])).all(), "array"


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()