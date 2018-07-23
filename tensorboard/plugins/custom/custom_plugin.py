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

import json
import textwrap
import threading
import time

# pylint: disable=g-bad-import-order
# Necessary for an internal test with special behavior for numpy.
import numpy as np
# pylint: enable=g-bad-import-order

import six
import tensorflow as tf
from werkzeug import wrappers

from tensorboard import plugin_util
from tensorboard.backend import http_util
from tensorboard.plugins import base_plugin
from tensorboard.plugins.custom import metadata

# HTTP routes
TAGS_ROUTE = '/tags'
CUSTOM_ROUTE = '/custom'


def text_array_to_html(text_arr):
  return plugin_util.markdown_to_safe_html(np.asscalar(text_arr))


def process_string_tensor_event(event):
  """Convert a TensorEvent into a JSON-compatible response."""
  string_arr = tf.make_ndarray(event.tensor_proto)
  html = text_array_to_html(string_arr)
  return {
      'wall_time': event.wall_time,
      'step': event.step,
      'text': html,
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

    # Cache the last result of index_impl() so that methods that depend on it
    # can return without blocking (while kicking off a background thread to
    # recompute the current index).
    self._index_cached = None

    # Lock that ensures that only one thread attempts to compute index_impl()
    # at a given time, since it's expensive.
    self._index_impl_lock = threading.Lock()

    # Pointer to the current thread computing index_impl(), if any.  This is
    # stored on CustomPlugin only to facilitate testing.
    self._index_impl_thread = None

  def is_active(self):
    """Determines whether this plugin is active.

    This plugin is only active if TensorBoard sampled any custom summaries.

    Returns:
      Whether this plugin is active.
    """
    return bool(self._multiplexer.PluginRunToTagToContent(metadata.PLUGIN_NAME))

  def _maybe_launch_index_impl_thread(self):
    """Attempts to launch a thread to compute index_impl().

    This may not launch a new thread if one is already running to compute
    index_impl(); in that case, this function is a no-op.
    """
    # Try to acquire the lock for computing index_impl(), without blocking.
    if self._index_impl_lock.acquire(False):
      # We got the lock. Start the thread, which will unlock the lock when done.
      self._index_impl_thread = threading.Thread(
          target=self._async_index_impl,
          name='CustomPluginIndexImplThread')
      self._index_impl_thread.start()

  def _async_index_impl(self):
    """Computes index_impl() asynchronously on a separate thread."""
    start = time.time()
    tf.logging.info('CustomPlugin computing index_impl() in a new thread')
    self._index_cached = self.index_impl()
    self._index_impl_thread = None
    self._index_impl_lock.release()
    elapsed = time.time() - start
    tf.logging.info(
        'CustomPlugin index_impl() thread ending after %0.3f sec', elapsed)

  def index_impl(self):
    run_to_series = self._fetch_run_to_series_from_multiplexer()

    # A previous system of collecting and serving text summaries involved
    # storing the tags of text summaries within tensors.json files. See if we
    # are currently using that system. We do not want to drop support for that
    # use case.
    name = 'tensorboard_custom'
    run_to_assets = self._multiplexer.PluginAssets(name)
    for run, assets in run_to_assets.items():
      if run in run_to_series:
        # When runs conflict, the summaries created via the new method override.
        continue

      if 'tensors.json' in assets:
        tensors_json = self._multiplexer.RetrievePluginAsset(
            run, name, 'tensors.json')
        tensors = json.loads(tensors_json)
        run_to_series[run] = tensors
      else:
        # The mapping should contain all runs among its keys.
        run_to_series[run] = []

    return run_to_series

  def _fetch_run_to_series_from_multiplexer(self):
    # TensorBoard is obtaining summaries related to the text plugin based on
    # SummaryMetadata stored within Value protos.
    mapping = self._multiplexer.PluginRunToTagToContent(
        metadata.PLUGIN_NAME)
    return {
        run: list(tag_to_content.keys())
        for (run, tag_to_content)
        in six.iteritems(mapping)
    }

  def tags_impl(self):
    # Recompute the index on demand whenever tags are requested, but do it
    # in a separate thread to avoid blocking.
    self._maybe_launch_index_impl_thread()

    # Use the cached index if present. If it's not, just return the result based
    # on data from the multiplexer, requiring no disk read.
    if self._index_cached:
      return self._index_cached
    else:
      return self._fetch_run_to_series_from_multiplexer()

  @wrappers.Request.application
  def tags_route(self, request):
    response = self.tags_impl()
    return http_util.Respond(request, response, 'application/json')

  def custom_impl(self, run, tag):
    try:
      text_events = self._multiplexer.Tensors(run, tag)
    except KeyError:
      text_events = []
    responses = [process_string_tensor_event(ev) for ev in text_events]
    return responses

  @wrappers.Request.application
  def custom_route(self, request):
    run = request.args.get('run')
    tag = request.args.get('tag')
    response = self.custom_impl(run, tag)
    return http_util.Respond(request, response, 'application/json')

  def get_plugin_apps(self):
    return {
        TAGS_ROUTE: self.tags_route,
        CUSTOM_ROUTE: self.custom_route,
    }
