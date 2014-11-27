__author__ = 'simon'
from abstract_update_method import AbstractUpdateMethod
import numpy as np


class Rprop(AbstractUpdateMethod):
    """
    Resilient propagation (Rprop) - see
    http://en.wikipedia.org/wiki/Rprop.
    """

    def __init__(self, layer_sizes, rate_pos=1.2, rate_neg=0.5,
                 init_step=0.0001, min_step=0.000001, max_step=50.):
        self.rate_pos = rate_pos
        self.rate_neg = rate_neg
        self.max_step = max_step
        self.min_step = min_step
        self.last_gradient = []
        self.step_size = []
        # initialize last gradient and step size for each layer and weight
        for layer in range(len(layer_sizes) - 1):
            init = np.zeros((layer_sizes[layer] + 1, layer_sizes[layer + 1]))
            self.last_gradient.append(init)
            # set step size to given initial step size
            self.step_size.append((init + 1) * init_step)

    def perform_update(self, weights, gradients, error):
        for i in range(len(weights)):
            last_gradient = self.last_gradient[i]
            step_size = self.step_size[i]
            gradient = gradients[i]
            weight = weights[i]

            # calculate the change in the gradient direction
            change = np.sign(gradient * last_gradient)

            # get the weights where:
            # -  change > 0 -> direction didn't change
            #    -  change < 0 -> direction changed
            #    - change == 0 -> one of the gradients is 0, do nothing
            greater_zero_idxs = np.where(change > 0)
            less_than_zero_idxs = np.where(change < 0)

            # direction didn't change -> increase step size
            # ( probably we are on a plateau where we can go faster )
            for idx in range(len(greater_zero_idxs[0])):
                r = greater_zero_idxs[0][idx]
                c = greater_zero_idxs[1][idx]
                step_size[r, c] = min(step_size[r, c] * self.rate_pos, self.max_step)

            # direction changed -> decrease step size
            # ( it seems we were too fast and jumped on the other side of
            # the minimum valley )
            for idx in range(len(less_than_zero_idxs[0])):
                r = less_than_zero_idxs[0][idx]
                c = less_than_zero_idxs[1][idx]
                step_size[r, c] = max(step_size[r, c] * self.rate_neg, self.min_step)

            change = -step_size * np.sign(gradient)
            weight += change
            last_gradient = np.copy(gradient)

        return weights