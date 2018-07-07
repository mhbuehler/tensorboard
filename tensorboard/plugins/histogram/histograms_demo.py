# -*- coding: utf-8 -*-
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Sample data exhibiting histogram summaries."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorboard.plugins.histogram import summary as histogram_summary

# Directory into which to write tensorboard data.
LOGDIR = '/tmp/histograms_demo'


def run_all(logdir, verbose=False):
  """Generate a bunch of histogram data, and write it to logdir."""
  del verbose

  tf.set_random_seed(0)

  k = tf.placeholder(tf.float32)

  # Make a normal distribution, with a shifting mean
  mean_moving_normal = tf.random_normal(shape=[10], mean=(5*k), stddev=1)
  # Record that distribution into a histogram summary
  tf.summary.histogram("normal/moving_mean", mean_moving_normal)

  # # Add a gamma distribution
  # gamma = tf.random_gamma(shape=[10], alpha=k)
  # histogram_summary.op("gamma", gamma,
  #                      description="A gamma distribution whose shape "
  #                                  "parameter, α, changes over time.")

  # # And a poisson distribution
  # poisson = tf.random_poisson(shape=[10], lam=k)
  # histogram_summary.op("poisson", poisson,
  #                      description="A Poisson distribution, which only "
  #                                  "takes on integer values.")

  # # And a uniform distribution
  # host_memory_uniform = tf.random_uniform(shape=[10], maxval=k*10)
  # histogram_summary.op("host_memory_uniform", host_memory_uniform,
  #                      description="A simple uniform distribution.")

  # device_memory_uniform = tf.random_uniform(shape=[10], maxval=k*10)
  # histogram_summary.op("device_memory_uniform", device_memory_uniform,
  #                      description="Device_memory uniform distribution")

  # Finally, combine everything together!
  all_distributions = [mean_moving_normal]
  total_memory_info = tf.concat(all_distributions, 0)
  histogram_summary.op("total_memory_info", total_memory_info,
                       description="Distributions of all combined, host_memory, device_memory etc.")

  summaries = tf.summary.merge_all()

  # Setup a session and summary writer
  sess = tf.Session()
  writer = tf.summary.FileWriter(logdir)

  # Setup a loop and write the summaries to disk
  N = 10
  for step in xrange(N):
  # k_val = N
  # print(k_val)
    summ = sess.run(summaries, feed_dict={k: step})
    writer.add_summary(summ)


def main(unused_argv):
  print('Running histograms demo. Output saving to %s.' % LOGDIR)
  run_all(LOGDIR)
  print('Done. Output saved to %s.' % LOGDIR)


if __name__ == '__main__':
  tf.app.run()
