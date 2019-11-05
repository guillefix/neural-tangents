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
from math import ceil


flags.DEFINE_float('learning_rate', 1.0,
                   'Learning rate to use during training.')
flags.DEFINE_integer('train_size', 1000,
                     'Dataset size to use for training.')
flags.DEFINE_integer('test_size', 100,
                     'Dataset size to use for testing.')
flags.DEFINE_float('train_time', 1000.0,
                   'Continuous time denoting duration of training.')
flags.DEFINE_integer('num_samples', 10,
                     'number of function samples')
flags.DEFINE_boolean('using_SGD', False,
                     'number of function samples')


FLAGS = flags.FLAGS

#%%


def main(unused_argv):
    using_SGD = FLAGS.using_SGD
    x_train,y_train, x_test, y_test = pickle.load(open("data.p","rb"))
    # x_train,y_train, x_test, y_test = pickle.load(open("data_10000.p","rb"))
    # y_train = 2*y_train-1
    # y_test = 2*y_test-1
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
    print("Getting NTK")
    sys.stdout.flush()
    #ntk = nt.batch(nt.empirical_ntk_fn(apply_fn), batch_size=4, device_count=0)
    #g_dd = ntk(x_train, None, params)
    #g_td = ntk(x_test, x_train, params)
    g_dd = pickle.load(open("ntk_train_"+str(FLAGS.train_size)+".p", "rb"))
    g_td = pickle.load(open("ntk_train_test_"+str(FLAGS.train_size)+".p", "rb"))
    # g_dd = pickle.load(open("ntk_train_10000.p", "rb"))
    # g_td = pickle.load(open("ntk_train_test_10000.p", "rb"))
    if not using_SGD:
        predictor = nt.predict.gradient_descent_mse(g_dd, y_train, g_td)

    g_dd.shape
    y_train.shape

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    print(rank)
    for i in range(FLAGS.num_samples):
        if i%(ceil(FLAGS.num_samples/100))==0:
            print(i)
            sys.stdout.flush()
        #reinitialize the network
        randnnn=numpy.random.random_integers(np.iinfo(np.int32).min,high=np.iinfo(np.int32).max,size=2)[0]
        key = random.PRNGKey(randnnn)
        _, params = init_fn(key, (-1, 784))

        # Get initial values of the network in function space.
        fx_train = apply_fn(params, x_train)
        fx_test = apply_fn(params, x_test)

        # Get predictions from analytic computation.
        #print('Computing analytic prediction.')
        if using_SGD:
            error = 1
            lr = 0.0001
            lr = nt.predict.max_learning_rate(g_dd)
            print(lr)
            lr *= 0.1
            ntk_train = g_dd.squeeze()
            ntk_train_test = g_td.squeeze()
            # print(np.matmul(ntk_train_test,(fx_train-fx_test)).shape)
            batch_size = FLAGS.train_size
            while error >= 1e-2:
                fx_test = fx_test - lr*np.matmul(ntk_train_test,(fx_train-y_train))/(2*batch_size)
                fx_train = fx_train - lr*np.matmul(ntk_train,(fx_train-y_train))/(2*batch_size)
                error = np.dot((fx_train-y_train).squeeze(),(fx_train-y_train).squeeze())/(2*batch_size)
                # print(error)
        else:
            fx_train, fx_test = predictor(FLAGS.train_time, fx_train, fx_test)
        # fx_test
        OUTPUT=fx_test>0.5
        #OUTPUT=fx_test>0
        OUTPUT=OUTPUT.astype(int)
        #print(np.transpose(OUTPUT))
        fun = ''.join([str(int(i)) for i in OUTPUT])
        fun
        TRUE_OUTPUT=y_test>0.5
        #TRUE_OUTPUT=y_test>0
        TRUE_OUTPUT=TRUE_OUTPUT.astype(int)
        #print(np.transpose(OUTPUT))
        ''.join([str(int(i)) for i in TRUE_OUTPUT])
        print("Generalization accuracy", np.sum(OUTPUT == TRUE_OUTPUT)/FLAGS.test_size)

        loss = lambda fx, y_hat: 0.5 * np.mean((fx - y_hat) ** 2)
        #util.print_summary('train', y_train, apply_fn(params, x_train), fx_train, loss)
        #util.print_summary('test', y_test, apply_fn(params, x_test), fx_test, loss)

        OUTPUT=fx_train>0.5
        OUTPUT=OUTPUT.astype(int)
        #print(np.transpose(OUTPUT))
        ''.join([str(int(i)) for i in OUTPUT])
        TRUE_OUTPUT=y_train>0.5
        TRUE_OUTPUT=OUTPUT.astype(int)
        #print(np.transpose(OUTPUT))
        ''.join([str(int(i)) for i in TRUE_OUTPUT])
        assert np.all(OUTPUT == TRUE_OUTPUT)

        file = open('data_{}_large.txt'.format(rank),'a')
        file.write(fun+'\n')
    file.close()

if __name__ == '__main__':
    app.run(main)
