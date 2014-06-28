'''
Created on Jan 11, 2014

@author: xapharius
'''
from datahandler.AbstractDataHandler import AbstractDataHandler
from datahandler.image.image_pre_processor import ImagePreProcessor
from datahandler.image.image_data_processor import ImageDataProcessor
from datahandler.image.image_stats import ImageStats
from datahandler.image.image_data_conf import ImageDataConf

class ImageDataHandler(AbstractDataHandler):
    '''
    DataHanlder for images.
    '''

    IMAGES_PER_MAP = 20

    def __init__(self):
        '''
        Constructor
        '''
        super(ImageDataHandler, self).__init__()

    def get_pre_processor(self):
        return ImagePreProcessor()

    def get_data_processor(self):
        return ImageDataProcessor()

    def get_configuration(self):
        return ImageDataConf(self.IMAGES_PER_MAP)

    def get_new_statistics(self):
        return ImageStats()
