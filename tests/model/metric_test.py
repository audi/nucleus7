# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

import tensorflow as tf

from nucleus7.test_utils.model_dummies import DummyMetric


class TestModelMetric(tf.test.TestCase):

    def test_call(self):
        tf.reset_default_graph()
        model_metric = DummyMetric(inbound_nodes=[], name='metric').build()
        model_metric.mode = 'train'
        inputs = {'predictions': tf.placeholder(tf.float32, shape=[None] * 3),
                  'labels': tf.placeholder(tf.float32, shape=[None] * 3)}

        metric = model_metric(**inputs)
        self.assertIsInstance(metric, dict)
        self.assertListEqual(['metric'], list(metric.keys()))

        tensor_names = {k: v.name for k, v in metric.items()}
        tensor_names_must = {k: "/".join([model_metric.name, k]) + ':0'
                             for k in metric}
        self.assertDictEqual(tensor_names_must,
                             tensor_names)
