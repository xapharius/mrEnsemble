'''
Created on Mar 12, 2014

@author: Simon
'''
from datahandler.AbstractPreProcessor import AbstractPreProcessor
import numpy as np
import sys

class NumericalPreProcessor(AbstractPreProcessor):
    '''
    Determines min, max and mean of a numerical data set.
    '''

    MIN, MAX, MEAN, NUM, VAR = 'min', 'max', 'mean', 'num', 'var'


    def calculate(self, data_set):
        min_val = np.min(data_set.inputs, axis=0).tolist()
        max_val = np.max(data_set.inputs, axis=0).tolist()
        mean_val = np.mean(data_set.inputs, axis=0).tolist()
        var_val = np.var(data_set.inputs, axis=0).tolist()
        num_val = data_set.nrObservations
        return { self.MIN: min_val, self.MAX: max_val, self.MEAN: mean_val, self.VAR: var_val, self.NUM: num_val }

    def aggregate(self, key, values):
        result = { self.MIN: 0, self.MAX: 0, self.MEAN: 0, self.NUM: 0, self.VAR: 0 }
        for stats in values:
            num = stats[self.NUM]
            _min = np.array(stats[self.MIN])
            _max = np.array(stats[self.MAX])
            mean = np.array(stats[self.MEAN])
            var = np.array(stats[self.VAR])
            if result[self.NUM] == 0:
                result[self.NUM] = num
                result[self.MIN] = _min
                result[self.MAX] = _max
                result[self.MEAN] = mean
                result[self.VAR] = var
            else:
                mins = np.vstack((result[self.MIN], _min))
                maxs = np.vstack((result[self.MAX], _max))
                
                result[self.MIN] = np.min(mins, axis=0)
                result[self.MAX] = np.max(maxs, axis=0)
                result[self.VAR] = (result[self.VAR]*result[self.NUM] + var*num)/(result[self.NUM] + num) + result[self.NUM]*num*np.power((mean-result[self.MEAN])/(num+result[self.NUM]), 2)
                result[self.MEAN] = (result[self.MEAN]*result[self.NUM] + mean*num)/(result[self.NUM] + num)
                result[self.NUM] += num
        sys.stderr.write('aggregate result: ' + str(result) + '\n')
        result[self.MIN] = result[self.MIN].tolist()
        result[self.MAX] = result[self.MAX].tolist()
        result[self.MEAN] = result[self.MEAN].tolist()
        result[self.VAR] = result[self.VAR].tolist()
        return result
