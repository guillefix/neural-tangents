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

"""Tests for `examples/function_space.py`."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from jax import test_util as jtu
from jax.config import config
from examples import infinite_fcn


config.parse_flags_with_absl()


class InfiniteFcnTest(jtu.JaxTestCase):

  def test_infinite_fcn(self):
    infinite_fcn.main(None)


if __name__ == '__main__':
  jtu.absltest.main()
