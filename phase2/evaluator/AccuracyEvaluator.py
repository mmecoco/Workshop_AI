from abc import ABCMeta, abstractmethod
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score

from Evaluator import Evaluator

class AccuracyEvaluator(Evaluator):

    @property
    def worst_score(self):
        return 0.0

    @property
    def mode(self):
        return 'max'

    def score(self, y_true, y_pred):
        return accuracy_score(y_true.argmax(axis=1), y_pred.argmax(axis=1))

    def is_better(self, curr, best, **kwargs):
        score_threshold = kwargs.pop('score_threshold', 1e-4)
        relative_eps = 1.0 + score_threshold
        return curr > best * relative_eps
