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

"""A set of utility operations for running examples.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import jax.numpy as np


def _accuracy(y_hat,y):
  """Compute the accuracy of the predictions with respect to one-hot labels."""
  # print(y)
  # print(y_hat)
  if y.shape[1] > 1:
      return np.mean(np.argmax(y, axis=1) == np.argmax(y_hat, axis=1))
  else:
      return np.mean(y == (y_hat>0.5))
      # return np.mean(y == np.sign(y_hat))


def print_summary(name, labels, net_p, lin_p, loss):
  """Print summary information comparing a network with its linearization."""
  print('\nEvaluating Network on {} data.'.format(name))
  print('---------------------------------------')
  print('Network Accuracy = {}'.format(_accuracy(net_p, labels)))
  print('Network Loss = {}'.format(loss(net_p, labels)))
  if lin_p is not None:
    print('Linearization Accuracy = {}'.format(_accuracy(lin_p, labels)))
    print('Linearization Loss = {}'.format(loss(lin_p, labels)))
    print('RMSE of predictions: {}'.format(
        np.sqrt(np.mean((net_p - lin_p) ** 2))))
  print('---------------------------------------')
