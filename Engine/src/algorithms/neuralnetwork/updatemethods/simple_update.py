__author__ = 'simon'
from abstract_update_method import AbstractUpdateMethod


class SimpleUpdate(AbstractUpdateMethod):
    """
    Simplest update method where the gradients are applied to the weights using
    a constant learning rate.
    """

    def __init__(self, learning_rate):
        """
        @param learning_rate: Step width for gradient descent
        """
        self.learning_rate = learning_rate

    def perform_update(self, weights, gradients, error):
        for i in range(len(weights)):
            weights[i] -= self.learning_rate * gradients[i]
        return weights
