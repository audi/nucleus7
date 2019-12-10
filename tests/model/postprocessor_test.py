# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

import tensorflow as tf

from nucleus7.test_utils.model_dummies import DummyPostProcessor


class TestModelPostprocessor(tf.test.TestCase):

    def test_call(self):
        tf.reset_default_graph()
        model_pp = DummyPostProcessor(inbound_nodes=[], name='pp').build()
        model_pp.mode = 'train'
        inputs = {'predictions': tf.placeholder(tf.int64, shape=[None] * 3)}
        results = model_pp(**inputs)
        self.assertIsInstance(results, dict)
        self.assertSetEqual({'predictions_pp'}, set(results.keys()))

        tensor_names = {k: v.name for k, v in results.items()}
        tensor_names_must = {k: "/".join([model_pp.name, k]) + ':0'
                             for k in results}
        self.assertDictEqual(tensor_names_must,
                             tensor_names)
