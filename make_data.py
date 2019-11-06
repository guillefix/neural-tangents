import sys
import os
import csv
from absl import app
from absl import flags
from jax.api import grad
from jax.api import jit
from jax.experimental import optimizers
import jax.numpy as np
import numpy
# from examples import datasets
# from examples import util
import pickle

flags.DEFINE_integer('train_size', 1000,
                     'Dataset size to use for training.')
flags.DEFINE_integer('test_size', 100,
                     'Dataset size to use for testing.')
flags.DEFINE_boolean('normalized', False,
                     'whether to rescale images to be between 0 and 255 or 0 and 1')

FLAGS = flags.FLAGS

def main(unused_argv):
  # Build data pipelines.
  print('Loading data.')
  sys.stdout.flush()
  #x_train, y_train, x_test, y_test = \
  #    datasets.mnist(FLAGS.train_size, FLAGS.test_size)

  #y_train=np.asarray([data_binariser(i) for i in y_train]).reshape(-1,1)
  #y_test=np.asarray([data_binariser(i) for i in y_test]).reshape(-1,1)

  from keras.datasets import mnist
  (X_train_full, y_train_full), (X_test_full, y_test_full) = mnist.load_data()

  #number_of_experiments = 10

  number_of_training_examples = FLAGS.train_size
  number_of_test_examples = FLAGS.test_size
  #small_number_of_test_examples = 100


  def data_binariser(i):
      if i%2 == 0:
          return 1
      return 0

  n = number_of_training_examples
  x_train = X_train_full[:n].reshape(n,784)
  y_train = np.asarray([data_binariser(i) for i in y_train_full[:n]]).reshape(n,1)

  n = number_of_test_examples
  x_test = X_test_full[:n].reshape(n,784)
  y_test = np.asarray([data_binariser(i) for i in y_test_full])[:n].reshape(n,1)
  if FLAGS.normalized:
    x_train = x_train/255.0
    x_test = x_test/255.0
  else:
    x_train = x_train
    x_test = x_test
  print(x_train.max())
  pickle.dump((x_train,y_train,x_test,y_test),open("data_"+str(number_of_training_examples)+".p","wb"))

if __name__ == '__main__':
  app.run(main)
