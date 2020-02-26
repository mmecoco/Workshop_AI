import os
import sys
import numpy as np
import tensorflow as tf
from imread import imread, imsave

sys.path.insert(0, "datasets")
sys.path.insert(0, "models")
sys.path.insert(0, "evaluator")
sys.path.insert(0, "optimizer")

from read_asirra_subset import read_asirra_subset
from DataSet import DataSet
from AlexNet import AlexNet as ConvNet
from AccuracyEvaluator import AccuracyEvaluator as Evaluator
from MomentumOptimizer import MomentumOptimizer as Optimizer

root_dir = os.path.join('../', 'data', 'asirra')    # FIXME
test_dir = os.path.join(root_dir, 'test')

X_test, y_test = read_asirra_subset(test_dir, one_hot=True)
test_set = DataSet(X_test, y_test)

print('Test set stats:')
print(test_set.images.shape)
print(test_set.images.min(), test_set.images.max())
print((test_set.labels[:, 1] == 0).sum(), (test_set.labels[:, 1] == 1).sum())


hp_d = dict()
image_mean = np.load('/tmp/asirra_mean.npy')
hp_d['image_mean'] = image_mean

hp_d['batch_size'] = 256
hp_d['augment_pred'] = True

graph = tf.get_default_graph()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

model = ConvNet([227, 227, 3], 2, **hp_d)
evaluator = Evaluator()
saver = tf.train.Saver()

sess = tf.Session(graph=graph, config=config)
saver.restore(sess, 'save/model.ckpt')
test_y_pred = model.predict(sess, test_set, **hp_d)
test_score = evaluator.score(test_set.labels, test_y_pred)

print('Test accuracy: {}'.format(test_score))
