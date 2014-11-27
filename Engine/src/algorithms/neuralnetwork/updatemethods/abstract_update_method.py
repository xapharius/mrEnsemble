__author__ = 'simon'
from abc import abstractmethod, ABCMeta


class AbstractUpdateMethod(object):
    """
    Base class for all gradient descent update methods.
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def perform_update(self, weights, gradients, error):
        """
        Apply weight update using given gradients and error.
        :param weights: Numpy array of weights that should be updated
        :param gradients: Gradients for the given weights
        :param error: Feedforward error
        @return: Updated weights
        """
        pass
