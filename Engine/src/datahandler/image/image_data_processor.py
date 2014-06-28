'''
Created on Jun 19, 2014

@author: Simon
'''
from datahandler.AbstractDataProcessor import AbstractDataProcessor
from datahandler.image.image_data_set import ImageDataSet

class ImageDataProcessor(AbstractDataProcessor):
    '''
    Data processor for images.
    '''


    def __init__(self):
        '''
        Constructor
        '''
        self.inputs = None
        self.targets = None

    def normalize_data(self, stats):
        '''
        TODO: not implemented
        '''
        pass

    def set_data(self, raw_data):
        '''
        @param rawData: value parameter for mrjob's map, list of numpy.ndarray 
        (images)
        '''
        self.inputs = raw_data
        
        super(ImageDataProcessor, self).set_data(raw_data)

    def get_data_set(self):
        '''
        @return: NumericalDataSet
        '''
        return ImageDataSet(self.inputs, self.targets)
