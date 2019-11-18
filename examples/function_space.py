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

from absl import app
from absl import flags
from jax import random
from jax.api import grad
from jax.api import jit
from jax.experimental import optimizers
import jax.numpy as np
import neural_tangents as nt
from neural_tangents import stax
from examples import datasets
from examples import util


flags.DEFINE_float('learning_rate', 1.0,
                   'Learning rate to use during training.')
flags.DEFINE_integer('train_size', 128,
                     'Dataset size to use for training.')
flags.DEFINE_integer('test_size', 128,
                     'Dataset size to use for testing.')
flags.DEFINE_float('train_time', 1000.0,
                   'Continuous time denoting duration of training.')


FLAGS = flags.FLAGS

FLAGS = {}
FLAGS["learning_rate"] = 1.0
FLAGS["train_size"] = 128
FLAGS["test_size"] = 128
FLAGS["train_time"] = 1000.0
class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)
FLAGS=Struct(**FLAGS)

#%%

def main(unused_argv):
    # Build data pipelines.
    print('Loading data.')
    x_train, y_train, x_test, y_test = \
      datasets.mnist(FLAGS.train_size, FLAGS.test_size)

    # Build the network
    init_fn, apply_fn, _ = stax.serial(
      stax.Dense(2048, 1., 0.05),
      # stax.Erf(),
      stax.Relu(),
      stax.Dense(2048, 1., 0.05),
      # stax.Erf(),
      stax.Relu(),
      stax.Dense(10, 1., 0.05))

    key = random.PRNGKey(0)
    _, params = init_fn(key, (-1, 784))

    # params

    # Create and initialize an optimizer.
    opt_init, opt_apply, get_params = optimizers.sgd(FLAGS.learning_rate)
    state = opt_init(params)
    # state


    # Create an mse loss function and a gradient function.
    loss = lambda fx, y_hat: 0.5 * np.mean((fx - y_hat) ** 2)
    grad_loss = jit(grad(lambda params, x, y: loss(apply_fn(params, x), y)))

    # Create an MSE predictor to solve the NTK equation in function space.
    ntk = nt.batch(nt.empirical_ntk_fn(apply_fn), batch_size=4, device_count=0)
    g_dd = ntk(x_train, None, params)
    g_td = ntk(x_test, x_train, params)
    predictor = nt.predict.gradient_descent_mse(g_dd, y_train, g_td)
    # g_dd.shape

    m = FLAGS.train_size
    print(m)
    n = m*10
    m_test = FLAGS.test_size
    n_test = m_test*10
    # g_td.shape
    # predictor
    # g_dd
    # type(g_dd)
    # g_dd.shape
    theta = g_dd.transpose((0,2,1,3)).reshape(n,n)
    theta_test = ntk(x_test, None, params).transpose((0,2,1,3)).reshape(n_test,n_test)
    theta_tilde = g_td.transpose((0,2,1,3)).reshape(n_test,n)
    #NNGP
    K = nt.empirical_nngp_fn(apply_fn)(x_train,None,params)
    K = np.kron(theta,np.eye(10))
    K_test = nt.empirical_nngp_fn(apply_fn)(x_test,None,params)
    K_test = np.kron(theta_test,np.eye(10))
    K_tilde = nt.empirical_nngp_fn(apply_fn)(x_test,x_train,params)
    K_tilde = np.kron(theta_tilde,np.eye(10))

    decay_matrix = np.eye(n)-scipy.linalg.expm(-t*theta)
    Sigma = K + np.matmul(decay_matrix, np.matmul(K, np.matmul(np.linalg.inv(theta), np.matmul(decay_matrix, theta))) - 2*K)

    # K.shape
    theta
    # alpha = np.matmul(np.linalg.inv(K),np.matmul(theta,np.linalg.inv(theta)))
    # y_train
    # alpha = np.matmul(np.linalg.inv(K), y_train.reshape(1280))
    # Sigma = K + np.matmul()
    # K = theta
    sigma_noise = 1.0
    Y = y_train.reshape(n)
    alpha = np.matmul(np.linalg.inv(np.eye(n)*(sigma_noise**2)+K),Y)
    # cov = np.linalg.inv(np.linalg.inv(K)+np.eye(n)/(sigma_noise**2))
    # covi = np.linalg.inv(cov)
    # covi = np.linalg.inv(K)+np.eye(n)/(sigma_noise**2)
    # print(covi)
    # np.linalg.det(K)
    eigs = np.linalg.eigh(K)[0]
    logdetcoviK = np.sum(np.log((eigs+sigma_noise**2) /sigma_noise**2))
    # coviK = np.matmul(covi,K)
    # coviK = np.eye(n) + K/(sigma_noise**2)
    # coviK
    # covi
    # np.linalg.det()
    # KL = 0.5*np.log(np.linalg.det(coviK)) + 0.5*np.trace(np.linalg.inv(coviK)) + 0.5*np.matmul(alpha.T,np.matmul(K,alpha)) - n/2
    KL = 0.5*logdetcoviK + 0.5*np.trace(np.linalg.inv(coviK)) + 0.5*np.matmul(alpha.T,np.matmul(K,alpha)) - n/2
    print(KL)

    delta = 2**-10
    bound = (KL+2*np.log(m)+1-np.log(delta))/m
    bound = 1-np.exp(-bound)
    bound
    print("bound", bound)

    import numpy
    bigK = numpy.zeros((n+n_test,n+n_test))
    bigK
    bigK[0:n,0:n] = K
    bigK[0:n,n:] = theta_tilde.T
    bigK[n:,0:n] = theta_tilde
    bigK[n:,n:] = theta_test
    init_ntk_f = numpy.random.multivariate_normal(np.zeros(n+n_test),bigK)
    fx_train = init_ntk_f[:n].reshape(m,10)
    fx_test = init_ntk_f[n:].reshape(m_test,10)

    # Get initial values of the network in function space.
    # fx_train = apply_fn(params, x_train)
    # fx_test = apply_fn(params, x_test)

    # Train the network.
    train_steps = int(FLAGS.train_time // FLAGS.learning_rate)
    print('Training for {} steps'.format(train_steps))

    for i in range(train_steps):
        params = get_params(state)
        state = opt_apply(i, grad_loss(params, x_train, y_train), state)

    # Get predictions from analytic computation.
    print('Computing analytic prediction.')
    # fx_train, fx_test = predictor(FLAGS.train_time, fx_train, fx_test)
    fx_train, fx_test = predictor(FLAGS.train_time, fx_train, fx_test)

    # Print out summary data comparing the linear / nonlinear model.
    util.print_summary('train', y_train, apply_fn(params, x_train), fx_train, loss)
    util.print_summary('test', y_test, apply_fn(params, x_test), fx_test, loss)

if __name__ == '__main__':
  app.run(main)
