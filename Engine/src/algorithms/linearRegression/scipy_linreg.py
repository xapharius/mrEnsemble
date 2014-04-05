'''
Created on Apr 4, 2014

@author: Simon
'''
from algorithms.AbstractAlgorithm import AbstractAlgorithm
from sklearn import linear_model
import numpy as np

class SciPyLinReg(AbstractAlgorithm):
    '''
    classdocs
    '''

    ORDINARY = 'ordinary'
    RIDGE    = 'ridge'
    LASSO    = 'lasso'

    def __init__(self, _type=ORDINARY, alpha=0.3):
        '''
        Constructor
        '''
        if _type == self.ORDINARY:
            self.lin_reg = linear_model.LinearRegression(fit_intercept=False)
        elif _type == self.RIDGE:
            self.lin_reg = linear_model.Ridge(alpha=alpha, fit_intercept=False)
        elif type == self.LASSO:
            self.lin_reg = linear_model.Lasso(alpha=alpha, fit_intercept=False)
        self.lin_reg.intercept_ = 0

    def train(self, data_set):
        self.lin_reg.fit(data_set.get_inputs(), data_set.get_targets().flatten())

    def predict(self, data_set):
        result = []
        for _input, _ in data_set.gen_observations():
            result.append(self.lin_reg.predict(_input))
        return result

    def get_params(self):
        return np.array(self.lin_reg.coef_)

    def set_params(self, parameters):
        self.lin_reg.coef_ = np.array(parameters).T
