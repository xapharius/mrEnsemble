'''
Created on Mar 18, 2015

@author: xapharius
'''

class ModelManager(object):
    '''
    classdocs
    '''


    def __init__(self, model, feature_selector):
        '''
        Constructor
        '''
        self.model = model
        self.feature_seletor = feature_selector
        self.training_data_statistics = None
        self.training_performance = None

    def _calc_data_statistics(self, dataset):
        '''
        Get statistics about the data (training) in order to understand the models capabilities
        '''
        #TODO: smart way of figuring statistics
        pass

    def _calc_performance(self, dataset):
        '''
        Calulates performance on dataset
        '''
        #TODO: some smart metrics here eg. partitioning hyperspace and get score for each cluster/bin
        return self.model.score(dataset.inputs, dataset.targets)

    def train(self, raw_data):
        '''
        Fit model to data as well as gather training data and performance statistics 
        '''
        dataset = self.feature_seletor.get_dataset(raw_data)
        self.training_data_statistics = self._calc_data_statistics(dataset)
        self.model.fit(dataset.inputs, dataset.targets)
        self.training_performance = self._calc_performance(dataset)

    def predict(self, raw_data):
        dataset = self.feature_seletor.get_dataset(raw_data)
        return self.model.predict(dataset.inputs)


