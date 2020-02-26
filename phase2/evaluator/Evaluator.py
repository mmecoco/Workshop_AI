from abc import ABCMeta, abstractmethod, abstractproperty
import numpy as np
import tensorflow as tf

class Evaluator(object):
    """Interface class for the evaluator"""

    @abstractproperty
    def worst_score(self):
        """
        returns the worst cost value
        """
        pass

    @abstractproperty
    def mode(self):
        """
        return "min" if the mode is set to min and "max" if it s a accuracy evaluator
        """
        pass

    @abstractmethod
    def score(self, y_true, y_pred):
        """
        :param y_true: np.ndarray, shape: (N, num_classes).
        :param y_pred: np.ndarray, shape: (N, num_classes).
        :return float.
        """
        pass

    @abstractmethod
    def is_better(self, curr, best, **kwargs):
        """
        :param curr: float, current score.
        :param best: float, max score.
        :return bool.
        """
        pass
