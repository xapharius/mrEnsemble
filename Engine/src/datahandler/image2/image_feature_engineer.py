'''
Created on Aug 1, 2015

@author: xapharius
'''
import numpy as np
import utils.numpyutils as nputils
from skimage.transform import resize
from datahandler.abstract_feature_engineer import AbstractFeatureEngineer
from datahandler.numerical.NumericalDataSet import NumericalDataSet


class ImageFeatureEngineer(AbstractFeatureEngineer):
    '''
    Engineering Images (eg adding noise or smth)
    '''

    def __init__(self, normalize=False, scale_to=None):
        self.normalize = normalize
        self.scale_to = scale_to
        self.number_of_features = None

    def get_dataset(self, raw_data, targets=None):
        '''
        @param raw_data: np.ndarray of matrices
        @param target: np.ndarray (n, 1)
        '''
        inputs = raw_data
        if self.normalize:
            inputs = [nputils.normalize_arr(img, -1, 1) for img in inputs]
        if self.scale_to is not None:
            inputs = [resize(img, self.scale_to) for img in inputs]
        return NumericalDataSet(inputs, targets)
