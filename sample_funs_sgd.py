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
import jax
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
flags.DEFINE_integer('train_steps', 1000,
                     'number of training steps')
flags.DEFINE_integer('batch_size', 32,
                     'number of examples in each training batch')
flags.DEFINE_string('loss', "mse",
                     'the loss function to use (mse/ce)')


FLAGS = flags.FLAGS

#%%

def main(unused_argv):
    loss = FLAGS.loss
    train_size = FLAGS.train_size
    x_train,y_train, x_test, y_test = pickle.load(open("data_"+str(train_size)+".p","rb"))
    print("Got data")
    sys.stdout.flush()

    # Build the network
    init_fn, apply_fn, _ = stax.serial(
      stax.Dense(2048, 1., 0.05),
      stax.Relu(),
      stax.Dense(1, 1., 0.05))

    opt_init, opt_apply, get_params = optimizers.sgd(FLAGS.learning_rate)

    # Create an mse loss function and a gradient function.
    if loss=="mse":
        loss = lambda fx, y_hat: 0.5 * np.mean((fx - y_hat) ** 2)
        decision_threshold = 0.5
    elif loss=="ce":
        loss = lambda fx, yhat: np.sum( (1-yhat)*np.log(1+np.exp(fx)) + (yhat)*(np.log(1+np.exp(fx))-fx) )
        decision_threshold = 0.0
    else:
        raise NotImplementedError()
    grad_loss = jit(grad(lambda params, x, y: loss(apply_fn(params, x), y)))

    batch_size = FLAGS.batch_size

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    print(rank)
    for i in range(FLAGS.num_samples):
        #reinitialize the network
        randnnn=numpy.random.random_integers(np.iinfo(np.int32).min,high=np.iinfo(np.int32).max,size=2)[0]
        key = random.PRNGKey(randnnn)
        _, params = init_fn(key, (-1, 784))
        state = opt_init(params)
        # Get initial values of the network in function space.
        fx_train = apply_fn(params, x_train)
        # fx_test = apply_fn(params, x_test)

        OUTPUT=(fx_train>0).astype(int)
        TRUE_OUTPUT=(y_train>0).astype(int)
        train_acc = np.sum(OUTPUT == TRUE_OUTPUT)/FLAGS.train_size
        while train_acc < 1.0:
            if batch_size != train_size:
                indices = numpy.random.choice(range(train_size), size=batch_size, replace=False)
            else:
                indices = np.array(list(range(train_size)))
            state = opt_apply(i, grad_loss(params, x_train[indices], y_train[indices]), state)
            params = get_params(state)
            fx_train = apply_fn(params, x_train)
            OUTPUT=(fx_train>decision_threshold).astype(int)
            TRUE_OUTPUT=(y_train>decision_threshold).astype(int)
            train_acc = np.sum(OUTPUT == TRUE_OUTPUT)/FLAGS.train_size
            # print(train_acc)

        fx_train = apply_fn(params, x_train)
        fx_test = apply_fn(params, x_test)

        OUTPUT=(fx_train>decision_threshold).astype(int)
        #print(np.transpose(OUTPUT))
        # ''.join([str(int(i)) for i in OUTPUT])
        TRUE_OUTPUT=(y_train>decision_threshold).astype(int)
        train_acc = np.sum(OUTPUT == TRUE_OUTPUT)/FLAGS.train_size
        print("Training accuracy", train_acc)
        assert train_acc == 1.0

        OUTPUT=fx_test>decision_threshold
        OUTPUT=OUTPUT.astype(int)
        fun = ''.join([str(int(i)) for i in OUTPUT])
        TRUE_OUTPUT=y_test>decision_threshold
        TRUE_OUTPUT=TRUE_OUTPUT.astype(int)
        ''.join([str(int(i)) for i in TRUE_OUTPUT])
        test_acc = np.sum(OUTPUT == TRUE_OUTPUT)/FLAGS.test_size
        print("Generalization accuracy", test_acc)

        file = open('data_{}_large.txt'.format(rank),'a')
        file.write(fun+'\n')
    file.close()

if __name__ == '__main__':
    app.run(main)
