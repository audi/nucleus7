# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

import numpy as np
import tensorflow as tf

from nucleus7.utils import tf_collections_utils


class TestTFCollectionsUtils(tf.test.TestCase):

    @staticmethod
    def _add_inputs_to_collections():
        np.random.seed(456854)
        tf.reset_default_graph()
        graph = tf.get_default_graph()

        collection_name = 'inputs'
        inputs = {'inp1': tf.constant(np.random.randn(10, 10)),
                  'inp2': tf.constant(np.random.rand(20)),
                  'inp_list': [tf.constant(np.random.rand(1, 10)),
                               tf.constant(np.random.rand(20, 20))],
                  'inp_dict': {'a': tf.constant(np.random.rand(20)),
                               'b': tf.constant(np.random.rand(20))}}
        tf_collections_utils.nested2collection(
            collection_name, inputs, graph=graph)
        return inputs, graph

    def test_nested2collection(self):
        inputs, graph = self._add_inputs_to_collections()
        inputs_must = {'inp1': inputs['inp1'],
                       'inp2': inputs['inp2'],
                       'inp_list//0': inputs['inp_list'][0],
                       'inp_list//1': inputs['inp_list'][1],
                       'inp_dict//a': inputs['inp_dict']['a'],
                       'inp_dict//b': inputs['inp_dict']['b']}

        all_collections = graph.get_all_collection_keys()
        all_collections_must = ['inputs::' + k for k in inputs_must.keys()]
        self.assertSetEqual(set(all_collections), set(all_collections_must))
        with self.test_session():
            for coll_name in all_collections:
                values = graph.get_collection(coll_name)[0]
                values = values.eval()
                values_must = inputs_must[coll_name.split(':')[-1]]
                values_must = values_must.eval()
                for v, v_must in zip(values, values_must):
                    self.assertAllEqual(v, v_must)

    def test_collection2nested(self):
        tf.reset_default_graph()
        inputs, graph = self._add_inputs_to_collections()
        inputs_retrieved = tf_collections_utils.collection2nested(
            'inputs', graph=graph)
        self.assertSetEqual(set(inputs.keys()), set(inputs_retrieved.keys()))
        with self.test_session():
            for k, values in inputs_retrieved.items():
                if isinstance(values, list):
                    values = [v.eval() for v in values]
                elif isinstance(values, dict):
                    values = {k: v.eval() for k, v in values.items()}
                else:
                    values = values.eval()
                values_must = inputs[k]
                if isinstance(values_must, list):
                    values_must = [v.eval() for v in values_must]
                elif isinstance(values, dict):
                    values_must = {k: v.eval() for k, v in values_must.items()}
                else:
                    values_must = values_must.eval()

                if isinstance(values_must, list):
                    for v, v_must in zip(values, values_must):
                        self.assertAllEqual(v, v_must)
                elif isinstance(values_must, dict):
                    self.assertSetEqual(set(values.keys()),
                                        set(values_must.keys()))
                    for key in values:
                        self.assertAllEqual(values[key], values_must[key])
                else:
                    self.assertAllEqual(values, values_must)
