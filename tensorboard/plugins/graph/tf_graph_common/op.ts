/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the 'License');
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an 'AS IS' BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

module tf.graph.op {
  export interface CompatibilityProvider {
    opValid: (opNode: OpNode) => boolean;
  }

  export class TpuCompatibilityProvider implements CompatibilityProvider {
    /**
     * Whitelist of current Tensorflow ops valid on the TPU
     */
    static readonly WHITELIST = [
      'Negative_28',
      'Multiply_52',
      'Multiply_12188',
      'Negative_12189',
      'Multiply_12190',
      'Multiply_12192',
      'Divide_12193',
      'Divide_12195',
      'Divide_12196',
      'Divide_12198',
    ];

    /**
     * Returns true if the node's inferred device is not the TPU.
     * Note that this is only a best-effort check.
     */
    private isNotTpuOp(opDevice: string): boolean {
      if (opDevice.toLowerCase().search('cpu:') != -1) {
        return true;
      }
      if (opDevice.toLowerCase().search('gpu:') != -1) {
        return true;
      }
      return (opDevice.toLowerCase().search('tpu') == -1);
    }
    opValid(opNode: OpNode): boolean {
      // Function library nodes are generally for internal use.
      if (opNode.name.search(FUNCTION_LIBRARY_NODE_PREFIX) == 0) {
        return true;
      }
      // Nodes that lack op types should be ignored.
      if (!opNode.op) {
        return true;
      }
      // If assigned a device that is not TPU-related assume op is valid.
      if (opNode.device && this.isNotTpuOp(opNode.device)) {
        return true;
      }
      // If assigned to the TPU_SYSTEM device, assume op is valid.
      if (opNode.device && opNode.device.search('TPU_SYSTEM') != -1) {
        return true;
      }
      return _.includes(TpuCompatibilityProvider.WHITELIST, opNode.op);
    }
  }

  export function checkOpsForCompatibility(
    graph: SlimGraph,
    provider: CompatibilityProvider) {
    if (provider === null) {
      throw new Error('Compatibility provider required, but got: ' + provider);
    }
    _.each(graph.nodes, (node) => {
      node.compatible = provider.opValid(node);
      _.each(node.inEmbeddings, (node) => {
        node.compatible = provider.opValid(node);
      });

      _.each(node.outEmbeddings, (node) => {
        node.compatible = provider.opValid(node);
      });
    });
  }
}  // close module tf.graph.op