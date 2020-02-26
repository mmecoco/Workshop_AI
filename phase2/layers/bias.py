import numpy as np
import tensorflow as tf

def bias_variable(shape, value=1.0):
    """
    Code une fonction qui prends la forme de la matrice et la valeur initial en parametre et 
    returnez une un tensorflow variable qui serais un Bias du modèle
    """
    biases = tf.get_variable("Remplit cette partie par toi meme")
    return biases
