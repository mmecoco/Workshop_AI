import numpy as np
import tensorflow as tf

def weight_variable(shape, stddev=0.01):
    """
    Codez une fonction qui prends la forme de la matrice et son stddev en parametre
    et returnez une tensorflow varaible qui correspond au weight du neural net
    """
    weights = tf.get_variable("A remplir vous meme")
    return weights
