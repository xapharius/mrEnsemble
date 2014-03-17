'''
Created on Jan 8, 2014

@author: xapharius
'''

from abc import ABCMeta, abstractmethod

class AbstractPreProcessor(object):
    '''
    Abstract class for a Preprocessor.
    Specifies which methods a Preprocessor of a Data Class should implement.
    The Preprocessor gets the necessary statistics for the DataProcessor through m/r on the dataset, 
    before the actual m/r of the engine.
    The statistics must then be passed to the DataProcessor through the engines Job Config
    '''
    __metaclass__ = ABCMeta

    def __init__(self):
        '''
        Constructor
        '''
        pass  

    #TODO: Send data to job_conf here or in Engine? 
#     @abstractmethod
#     def get_statistics(self, dataSource):
#         '''
#         Creates dictionary with statistics of the dataset
#         Dictionary is constructed through map/reduce as the dataset is stored in HDFS
#         Statistics are specific for each Dataclass (e.g images, timeseries etc)
#         @param dataSource: location on HDFS of data that wants to be analysed 
#         @return: Dictionary of collected statistics
#         '''
#         pass

    @abstractmethod
    def calculate(self, data_set):
        '''
        Actual pre-processing happens here. I.e. should determine necessary 
        information from the given values, like max, min, avg, ...
        This is basically the Map step of the M/R pre-processing and should 
        return a dictionary of the different results, e.g result['min'] = 0.
        '''
        pass

    @abstractmethod
    def aggregate(self, key, values):
        '''
        Results from different pre-processor's 'calculate' should be merged here
        to give the overall result of the pre-processing. Given key is one of
        the keys returned in 'calculate'. It is probably a good idea to respond
        to different keys with a different behavior.
        This is basically the Reduce step of the M/R pre-processing.
        '''
        pass
