import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers.schedules import LearningRateSchedule, ExponentialDecay

from collections import Counter
from matplotlib import pyplot as plt
from sklearn.utils.class_weight import compute_sample_weight


class WarmUpAndDecayScheduler(LearningRateSchedule):
    def __init__(self, initial_learning_rate, warmup_steps, decay_steps, decay_rate):
        super(WarmUpAndDecayScheduler, self).__init__()
        self.initial_learning_rate = initial_learning_rate
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.decay_scheduler = ExponentialDecay(initial_learning_rate, decay_steps=decay_steps, decay_rate=decay_rate)
        if self.decay_steps <= 0: raise ValueError(f'Argument `decay_steps` must be > 0. Received: decay_steps={self.decay_steps}')

    def __call__(self, step):
        warmup_lr = self.initial_learning_rate * (step / self.warmup_steps)
        # decay_lr = tf.cast(self.decay_scheduler(step), tf.float32)
        decay_lr = self.initial_learning_rate * self.decay_rate ** ((step - self.warmup_steps) / self.decay_steps)
        return tf.cond(step < self.warmup_steps, lambda: warmup_lr, lambda: decay_lr)


def argmax_label_matrix(label_matrix, multi_branch=False):
    if not isinstance(label_matrix, np.ndarray): label_matrix = np.array(label_matrix)
    if not multi_branch: 
        y = label_matrix.reshape(len(label_matrix), -1, 4)
        return np.argmax(y, axis=-1)
    return np.argmax(label_matrix, axis=-1).T


def compute_class_weight(label_matrix, multi_branch=False, use_sample_weight=False):
    argmax_labels = argmax_label_matrix(label_matrix, multi_branch)
    if use_sample_weight: return compute_sample_weight('balanced', y=argmax_labels)
    counter = Counter(argmax_labels.reshape(-1))
    majority = max(counter.values())
    return {cls: float(majority / count) for cls, count in counter.items()}


def plot_training_history(history, figsize=(15, 5)):
    plt.figure(figsize=figsize)
    plt.plot(history['loss'], linestyle='solid', marker='o', color='crimson', label='Train')
    plt.plot(history['val_loss'], linestyle='solid', marker='o', color='dodgerblue', label='Validation')
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Loss',fontsize=14)
    plt.title('Loss', fontsize=15)
    plt.legend(loc='best')
    plt.show()