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
"""The TensorBoard Custom plugin."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=g-bad-import-order
# Necessary for an internal test with special behavior for numpy.
import numpy as np
# pylint: enable=g-bad-import-order

import six
import tensorflow as tf
from werkzeug import wrappers

from tensorboard.backend import http_util
from tensorboard.plugins import base_plugin
from tensorboard.plugins.custom import metadata

# HTTP routes
TAGS_ROUTE = '/tags'
CUSTOM_ROUTE = '/custom'


def process_string_tensor_event(event):
  """Convert a TensorEvent into a JSON-compatible response."""
  string_arr = tf.make_ndarray(event.tensor_proto)
  text = string_arr.astype(np.dtype(str)).tostring()
  return {
      'wall_time': event.wall_time,
      'step': event.step,
      'text': text,
  }


class CustomPlugin(base_plugin.TBPlugin):
  """Custom Plugin for TensorBoard."""

  plugin_name = metadata.PLUGIN_NAME

  def __init__(self, context):
    """Instantiates CustomPlugin via TensorBoard core.

    Args:
      context: A base_plugin.TBContext instance.
    """
    self._multiplexer = context.multiplexer

  def is_active(self):
    """Determines whether this plugin is active.
    This plugin is only active if TensorBoard sampled any summaries
    relevant to the custom plugin.
    Returns:
      Whether this plugin is active.
    """
    all_runs = self._multiplexer.PluginRunToTagToContent(
        metadata.PLUGIN_NAME)

    # The plugin is active if any of the runs has a tag relevant
    # to the plugin.
    return bool(self._multiplexer and any(six.itervalues(all_runs)))

  @wrappers.Request.application
  def tags_route(self, request):
    """A route (HTTP handler) that returns a response with tags.
    Returns:
      A response that contains a JSON object. The keys of the object
      are all the runs. Each run is mapped to a (potentially empty)
      list of all tags that are relevant to this plugin.
    """
    # This is a dictionary mapping from run to (tag to string content).
    # To be clear, the values of the dictionary are dictionaries.
    all_runs = self._multiplexer.PluginRunToTagToContent(
        metadata.PLUGIN_NAME)

    # tagToContent is itself a dictionary mapping tag name to string
    # content. We retrieve the keys of that dictionary to obtain a
    # list of tags associated with each run.
    response = {
        run: list(tagToContent.keys())
        for (run, tagToContent) in all_runs.items()
    }
    return http_util.Respond(request, response, 'application/json')

  @wrappers.Request.application
  def custom_route(self, request):
    """A route that returns the greetings associated with a tag.
    Returns:
      A JSON list of greetings associated with the run and tag
      combination.
    """
    run = request.args.get('run')
    tag = request.args.get('tag')

    # We fetch all the tensor events that contain greetings.
    tensor_events = self._multiplexer.Tensors(run, tag)

    # We convert the tensor data to text.
    response = [process_string_tensor_event(ev) for
                ev in tensor_events]
    return http_util.Respond(request, response, 'application/json')


  def get_plugin_apps(self):
    return {
        TAGS_ROUTE: self.tags_route,
        CUSTOM_ROUTE: self.custom_route,
    }
