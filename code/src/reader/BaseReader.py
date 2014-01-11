from abc import ABCMeta, abstractmethod

class BaseReader(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_data(self):
        pass

    @abstractmethod
    def set_data_source(self, source):
        pass