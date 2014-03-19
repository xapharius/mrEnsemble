'''
Created on Mar 12, 2014

@author: Simon
'''
from datahandler.AbstractPreProcessor import AbstractPreProcessor
import numpy as np
import sys

class NumericalPreProcessor(AbstractPreProcessor):
    '''
    Determines min, max, mean and variance of a numerical data set.
    '''

    DATA, LABEL                 = 'data', 'label'
    MIN, MAX, MEAN, NUM, VAR    = 'min', 'max', 'mean', 'num', 'var'


    def calculate(self, data_set):
        num = data_set.nrObservations

        data_min  = np.min(data_set.inputs, axis=0).tolist()
        data_max  = np.max(data_set.inputs, axis=0).tolist()
        data_mean = np.mean(data_set.inputs, axis=0).tolist()
        data_var  = np.var(data_set.inputs, axis=0).tolist()

        label_min  = np.min(data_set.targets, axis=0)[0]
        label_max  = np.max(data_set.targets, axis=0)[0]
        label_mean = np.mean(data_set.targets, axis=0)[0]
        label_var  = np.var(data_set.targets, axis=0)[0]

        return { 
                    self.NUM: num,
                    self.DATA: { 
                         self.MIN:  data_min,
                         self.MAX:  data_max,
                        self.MEAN:  data_mean,
                         self.VAR:  data_var
                    },
                    self.LABEL: { 
                         self.MIN:  label_min,
                         self.MAX:  label_max,
                        self.MEAN:  label_mean,
                         self.VAR:  label_var
                    }
                }

    def aggregate(self, key, values):
        result = { 
                    self.NUM: 0,
                    self.DATA: { 
                         self.MIN: 0,
                         self.MAX: 0,
                        self.MEAN: 0,
                         self.VAR: 0
                    },
                    self.LABEL: { 
                         self.MIN: 0,
                         self.MAX: 0,
                        self.MEAN: 0,
                         self.VAR: 0
                    }
                }
        for stats in values:
            num        = stats[self.NUM]
            data_min   = np.array(stats[self.DATA][self.MIN])
            data_max   = np.array(stats[self.DATA][self.MAX])
            data_mean  = np.array(stats[self.DATA][self.MEAN])
            data_var   = np.array(stats[self.DATA][self.VAR])
            label_min  = stats[self.LABEL][self.MIN]
            label_max  = stats[self.LABEL][self.MAX]
            label_mean = stats[self.LABEL][self.MEAN]
            label_var  = stats[self.LABEL][self.VAR]

            # when this is the first statistic just copy all values
            # otherwise combine the actual result with the statistic
            if result[self.NUM] == 0:
                result[self.NUM]              = num
                result[self.DATA][self.MIN]   = data_min
                result[self.DATA][self.MAX]   = data_max
                result[self.DATA][self.MEAN]  = data_mean
                result[self.DATA][self.VAR]   = data_var
                result[self.LABEL][self.MIN]  = label_min
                result[self.LABEL][self.MAX]  = label_max
                result[self.LABEL][self.MEAN] = label_mean
                result[self.LABEL][self.VAR]  = label_var
            else:
                data_mins = np.vstack((result[self.DATA][self.MIN], data_min))
                data_maxs = np.vstack((result[self.DATA][self.MAX], data_max))

                result[self.DATA][self.MIN]  = np.min(data_mins, axis=0)
                result[self.DATA][self.MAX]  = np.max(data_maxs, axis=0)
                result[self.DATA][self.VAR]  = (result[self.DATA][self.VAR]*result[self.NUM] + data_var*num)/(result[self.NUM] + num) + result[self.NUM]*num*np.power((data_mean-result[self.DATA][self.MEAN])/(num+result[self.NUM]), 2)
                result[self.DATA][self.MEAN] = (result[self.DATA][self.MEAN]*result[self.NUM] + data_mean*num)/(result[self.NUM] + num)

                result[self.LABEL][self.MIN]  = min([result[self.LABEL][self.MIN], label_min])
                result[self.LABEL][self.MAX]  = max([result[self.LABEL][self.MAX], label_max])
                result[self.LABEL][self.VAR]  = (result[self.LABEL][self.VAR]*result[self.NUM] + label_var*num)/(result[self.NUM] + num) + result[self.NUM]*num*np.power((label_mean-result[self.LABEL][self.MEAN])/(num+result[self.NUM]), 2)
                result[self.LABEL][self.MEAN] = (result[self.LABEL][self.MEAN]*result[self.NUM] + label_mean*num)/(result[self.NUM] + num)

                result[self.NUM] += num

        sys.stderr.write('aggregate result: ' + str(result) + '\n')
        # convert numpy arrays to list for transport
        result[self.DATA][self.MIN]  = result[self.DATA][self.MIN].tolist()
        result[self.DATA][self.MAX]  = result[self.DATA][self.MAX].tolist()
        result[self.DATA][self.MEAN] = result[self.DATA][self.MEAN].tolist()
        result[self.DATA][self.VAR]  = result[self.DATA][self.VAR].tolist()
        return result
