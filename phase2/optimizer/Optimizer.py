from abc import ABCMeta, abstractmethod
import numpy as np
import tensorflow as tf
import time
from tqdm import tqdm
import pickle as pkl
import matplotlib.pyplot as plt
import os

def plot_learning_curve(exp_idx, step_losses, step_scores, eval_scores=None,
                        mode='max', img_dir='.'):
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    axes[0].plot(np.arange(1, len(step_losses)+1), step_losses, marker='')
    axes[0].set_ylabel('loss')
    axes[0].set_xlabel('Number of iterations')
    axes[1].plot(np.arange(1, len(step_scores)+1), step_scores, color='b', marker='')
    if eval_scores is not None:
        axes[1].plot(np.arange(1, len(eval_scores)+1), eval_scores, color='r', marker='')
    if mode == 'max':
        axes[1].set_ylim(0.5, 1.0)
    else:    # mode == 'min'
        axes[1].set_ylim(0.0, 0.5)
    axes[1].set_ylabel('Accuracy')
    axes[1].set_xlabel('Number of epochs')

    # Save plot as image file
    plot_img_filename = 'learning_curve-result{}.svg'.format(exp_idx)
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    fig.savefig(os.path.join(img_dir, plot_img_filename))

    # Save details as pkl file
    pkl_filename = 'learning_curve-result{}.pkl'.format(exp_idx)
    with open(os.path.join(img_dir, pkl_filename), 'wb') as fo:
        pkl.dump([step_losses, step_scores, eval_scores], fo)


class Optimizer(object):

    def __init__(self, model, train_set, evaluator, val_set=None, **kwargs):
        self.model = model
        self.train_set = train_set
        self.evaluator = evaluator
        self.val_set = val_set

        self.batch_size = kwargs.pop('batch_size', 256)
        self.num_epochs = kwargs.pop('num_epochs', 320)
        self.init_learning_rate = kwargs.pop('init_learning_rate', 0.01)

        self.learning_rate_placeholder = tf.placeholder(tf.float32)
        self.optimize = self._optimize_op()

        self._reset()

    def _reset(self):
        self.curr_epoch = 1
        self.num_bad_epochs = 0
        self.best_score = self.evaluator.worst_score
        self.curr_learning_rate = self.init_learning_rate

    @abstractmethod
    def _optimize_op(self, **kwargs):
        """
        Doit implementer le tf.train.Optimizer.minimize Op.
        """
        pass

    @abstractmethod
    def _update_learning_rate(self, **kwargs):
        """
        Method qui permet d'update le learning rate si necessaire
        """
        pass

    def _step(self, sess, **kwargs):
        """
        :param sess: tf.Session.
        :param kwargs: dict, hyperparameter du training phase
            - augment_train: bool, true si on veux multiplier les datasets.
        :return loss: float, valeur du cost fonction au 1er tour.
                y_true: np.ndarray, answer of the dataset.
                y_pred: np.ndarray, prediction of the model.
        """
        augment_train = kwargs.pop('augment_train', True)

        X, y_true = self.train_set.next_batch(self.batch_size, shuffle=True,
                                              augment=augment_train, is_train=True)

        _, loss, y_pred = \
            sess.run([self.optimize, self.model.loss, self.model.pred],
                     feed_dict={self.model.X: X, self.model.y: y_true,
                                self.model.is_train: True,
                                self.learning_rate_placeholder: self.curr_learning_rate})

        return loss, y_true, y_pred

    def train(self, sess, save_dir='/tmp', details=False, verbose=True, **kwargs):
        """
        Execute the optimizer and train the model
        :param sess: tf.Session.
        :param save_dir: str, save directory path
        :param details: bool, true if more details return is needed.
        :param verbose: bool, verbose mode.
        :param kwargs: dict, hyperparameters.
        :return train_results: dict, a python dictionary that contais the required details.
        """
        saver = tf.train.Saver()
        try:
            if (save_dir == " /tmp"):
                sess.run(tf.global_variables_initializer())
            else:
                saver.restore(sess, save_dir+"/model.ckpt")
        except:
            sess.run(tf.global_variables_initializer())

        train_results = dict()
        train_size = self.train_set.num_examples
        num_steps_per_epoch = train_size // self.batch_size
        num_steps = self.num_epochs * num_steps_per_epoch

        if verbose:
            print('Running training loop...')
            print('Number of training iterations: {}'.format(num_steps))

        step_losses, step_scores, eval_scores = [], [], []
        start_time = time.time()

        for i in tqdm(range(num_steps)):
            step_loss, step_y_true, step_y_pred = self._step(sess, **kwargs)
            step_losses.append(step_loss)

            if (i+1) % num_steps_per_epoch == 0:
                step_score = self.evaluator.score(step_y_true, step_y_pred)
                step_scores.append(step_score)

                if self.val_set is not None:
                    eval_y_pred = self.model.predict(sess, self.val_set, verbose=False, **kwargs)
                    eval_score = self.evaluator.score(self.val_set.labels, eval_y_pred)
                    eval_scores.append(eval_score)

                    if verbose:
                        print('[epoch {}]\tloss: {:.6f} |Train score: {:.6f} |Eval score: {:.6f} |lr: {:.6f}'\
                              .format(self.curr_epoch, step_loss, step_score, eval_score, self.curr_learning_rate))
                        plot_learning_curve(-1, step_losses, step_scores, eval_scores=eval_scores,
                                            mode=self.evaluator.mode, img_dir=save_dir)
                    curr_score = eval_score

                else:
                    if verbose:
                        print('[epoch {}]\tloss: {} |Train score: {:.6f} |lr: {:.6f}'\
                              .format(self.curr_epoch, step_loss, step_score, self.curr_learning_rate))
                        plot_learning_curve(-1, step_losses, step_scores, eval_scores=None,
                                            mode=self.evaluator.mode, img_dir=save_dir)
                    curr_score = step_score

                if self.evaluator.is_better(curr_score, self.best_score, **kwargs):
                    self.best_score = curr_score
                    self.num_bad_epochs = 0
                    saver.save(sess, os.path.join(save_dir, 'model.ckpt'))
                else:
                    self.num_bad_epochs += 1

                self._update_learning_rate(**kwargs)
                self.curr_epoch += 1

        if verbose:
            print('Total training time(sec): {}'.format(time.time() - start_time))
            print('Best {} score: {}'.format('evaluation' if eval else 'training',
                                             self.best_score))
        print('Done.')

        if details:
            train_results['step_losses'] = step_losses    # (num_iterations)
            train_results['step_scores'] = step_scores    # (num_epochs)
            if self.val_set is not None:
                train_results['eval_scores'] = eval_scores    # (num_epochs)

            return train_results
