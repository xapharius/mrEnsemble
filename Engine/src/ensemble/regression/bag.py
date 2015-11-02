'''
Created on Mar 22, 2015

@author: xapharius
'''
from ensemble.abstract_ensemble import AbstractEnsemble

class Bag(AbstractEnsemble):
    '''
    classdocs
    '''

    def __init__(self, list_of_managers):
        '''
        Constructor
        '''
        self.managers = list_of_managers

    def predict(self, raw_data):
        predictions = [model.predict(raw_data) for model in self.managers]
        return sum(predictions)/float(len(predictions))