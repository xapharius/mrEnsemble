__author__ = 'simon'
from abstract_update_method import AbstractUpdateMethod
import numpy as np


class IRpropPlus(AbstractUpdateMethod):
    """
    Variation of Rprop - see http://en.wikipedia.org/wiki/Rprop.
    """

    def __init__(self, layer_sizes, rate_pos=1.2, rate_neg=0.5,
                  init_step=0.0001, min_step=0.000001, max_step=50.):
        self.rate_pos = rate_pos
        self.rate_neg = rate_neg
        self.max_step = max_step
        self.min_step = min_step
        self.last_gradient = []
        self.last_error = -1
        self.step_size = []
        self.last_change = []
        # initialize last gradient and step size for each layer and weight
        for layer in range(len(layer_sizes) - 1):
            init = np.zeros(
                (layer_sizes[layer] + 1, layer_sizes[layer + 1]))
            self.last_gradient.append(init)
            self.last_change.append(init)
            # set step size to given initial step size
            self.step_size.append((init + 1) * init_step)

    def perform_update(self, weights, gradients, error):
        if self.last_error == -1:
            self.last_error = error
        for i in range(len(weights)):
            last_gradient = self.last_gradient[i]
            last_change = self.last_change[i]
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
            equals_zero_idxs = np.where(change == 0)

            # direction didn't change -> increase step size
            # (probably we are on a plateau where we can go faster)
            for idx in range(len(greater_zero_idxs[0])):
                r = greater_zero_idxs[0][idx]
                c = greater_zero_idxs[1][idx]
                step_size[r, c] = min(step_size[r, c] * self.rate_pos,
                                         self.max_step)
                change = -step_size[r, c] * np.sign(gradient[r, c])
                weight[r, c] += change
                last_change[r, c] = change
                last_gradient[r, c] = gradient[r, c]

            # direction changed -> decrease step size
            # (it seems we were too fast and jumped on the other side of
            # the minimum valley)
            for idx in range(len(less_than_zero_idxs[0])):
                r = less_than_zero_idxs[0][idx]
                c = less_than_zero_idxs[1][idx]
                # decrease step size
                step_size[r, c] = max(step_size[r, c] * self.rate_neg,
                                         self.min_step)
                # if the error increased, revert last change
                if error > self.last_error:
                    weight[r, c] -= last_change[r, c]
                last_gradient[r, c] = 0

            # either the last or the current gradient is zero, that's not too
            # bad, so let's just go on and keep the step size
            for idx in range(len(equals_zero_idxs[0])):
                r = equals_zero_idxs[0][idx]
                c = equals_zero_idxs[1][idx]
                change = -step_size[r, c] * np.sign(gradient[r, c])
                weight[r, c] += change
                last_change[r, c] = change
                last_gradient[r, c] = gradient[r, c]

        self.last_error = error
        return weights