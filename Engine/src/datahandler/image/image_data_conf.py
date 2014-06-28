'''
Created on Jun 26, 2014

@author: Simon
'''
from datahandler.abstract_data_conf import AbstractDataConf
from datahandler.image.image_pre_proc_conf import ImagePreProcConf
from datahandler.image.image_training_conf import ImageTrainingConf

class ImageDataConf(AbstractDataConf):
    '''
    Configuration to control data splitting and serialization.
    '''

    def __init__(self, files_per_map):
        self.files_per_map = files_per_map

    def get_pre_proc_conf(self):
        '''
        @return: ImagePreProcConf
        '''
        return ImagePreProcConf(self.files_per_map)
    
    def get_training_conf(self):
        '''
        @return: ImageTrainingConf
        '''
        return ImageTrainingConf(self.files_per_map)
    
    def get_validation_conf(self):
        '''
        @return: ImageTrainingConf
        '''
        return ImageTrainingConf(self.files_per_map)
