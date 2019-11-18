# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""An example comparing training a neural network with the NTK dynamics.

In this example, we train a neural network on a small subset of MNIST using an
MSE loss and SGD. We compare this training with the analytic function space
prediction using the NTK. Data is loaded using tensorflow datasets.
"""
import sys
import os
import csv
from absl import app
from absl import flags
from jax import random
from jax.api import grad
from jax.api import jit
from jax.experimental import optimizers
import jax.numpy as np
import numpy
import neural_tangents as nt
from neural_tangents import stax
from examples import datasets
from examples import util
import pickle


flags.DEFINE_float('learning_rate', 1.0,
                   'Learning rate to use during training.')
flags.DEFINE_integer('train_size', 1000,
                     'Dataset size to use for training.')
flags.DEFINE_integer('test_size', 100,
                     'Dataset size to use for testing.')
flags.DEFINE_float('train_time', 1000.0,
                   'Continuous time denoting duration of training.')
flags.DEFINE_integer('num_samples', 100,
                     'number of function samples')


FLAGS = flags.FLAGS


def main(unused_argv):

  train_size = FLAGS.train_size
  x_train,y_train, x_test, y_test = pickle.load(open("data_"+str(train_size)+".p","rb"))
  print("Got data")
  sys.stdout.flush()

  # Build the network
  init_fn, apply_fn, _ = stax.serial(
      stax.Dense(2048, 1., 0.05),
      # stax.Erf(),
      stax.Relu(),
      stax.Dense(1, 1., 0.05))

  # initialize the network first time, to compute NTK
  randnnn=numpy.random.random_integers(np.iinfo(np.int32).min,high=np.iinfo(np.int32).max,size=2)[0]
  key = random.PRNGKey(randnnn)
  _, params = init_fn(key, (-1, 784))

  # Create an MSE predictor to solve the NTK equation in function space.
  # we assume that the NTK is approximately the same for any sample of parameters (true in the limit of infinite width)

  print("Making NTK")
  sys.stdout.flush()
  ntk = nt.batch(nt.empirical_ntk_fn(apply_fn), batch_size=4, device_count=1)
  g_dd = ntk(x_train, None, params)
  pickle.dump(g_dd,open("ntk_train_"+str(FLAGS.train_size)+".p","wb"))
  g_td = ntk(x_test, x_train, params)
  pickle.dump(g_td,open("ntk_train_test_"+str(FLAGS.train_size)+".p","wb"))
  predictor = nt.predict.gradient_descent_mse(g_dd, y_train, g_td)
  # pickle.dump(predictor,open("ntk_predictor.p","wb"))

if __name__ == '__main__':
  app.run(main)
