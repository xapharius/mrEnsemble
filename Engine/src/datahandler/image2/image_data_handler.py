'''
Created on Aug 1, 2015

@author: xapharius
'''

from datahandler.image2.image_feature_engineer import ImageFeatureEngineer
import random

class ImageDataHandler(object):
    '''
    Factory for NumericalFeatureSelectors
    '''


    def __init__(self):
        '''
        Factory Settings
        #TODO: maybe random scaling?
        #TODO: add maybe random noise and duplicate
        '''
        pass

    def get_feature_engineer(self):
        return ImageFeatureEngineer()