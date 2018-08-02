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
"""Sample text summaries exhibiting all the text plugin features."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorboard.plugins.custom import summary

# Directory into which to write tensorboard data.
LOGDIR = '/tmp/custom_demo'


def run(logdir, run_name, characters, extra_character):
  tf.reset_default_graph()

  input_ = tf.placeholder(tf.string)

  summary.op("greetings", input_)

  all_summaries = tf.summary.merge_all()

  with tf.Session() as sess:
    writer = tf.summary.FileWriter(os.path.join(logdir, run_name))
    for character in characters:
      s = sess.run(all_summaries, feed_dict={input_: character})
      writer.add_summary(s)
    # Demonstrate that we can also add summaries without using the
    # TensorFlow session or graph.
    s = summary.pb("greetings", extra_character)
    writer.add_summary(s)
      
    writer.close()

def run_all(logdir, unused_verbose=False):
  run(logdir, "steven_universe", ["Garnet", "Amethyst", "Pearl"], "Steven")
  run(logdir, "futurama", ["Fry", "Bender", "Leela"],
      "Lrrr, ruler of the planet Omicron Persei 8")


def main(unused_argv):
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.logging.info('Saving output to %s.' % LOGDIR)
  run_all(LOGDIR)
  tf.logging.info('Done. Output saved to %s.' % LOGDIR)


if __name__ == '__main__':
  tf.app.run()
