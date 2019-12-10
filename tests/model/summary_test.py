# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

import tensorflow as tf

from nucleus7.test_utils.model_dummies import DummySummary


class TestModelSummary(tf.test.TestCase):

    def test_call(self):
        tf.reset_default_graph()
        model_summary = DummySummary(inbound_nodes=[], name='summary').build()
        model_summary.mode = 'train'
        inputs = {'predictions': tf.placeholder(tf.int64, shape=[None] * 3),
                  'labels': tf.placeholder(tf.int64, shape=[None] * 3)}

        summary = model_summary(**inputs)
        self.assertIsInstance(summary, dict)
        self.assertSetEqual(
            {'image_predictions_class', 'image_labels_class'},
            set(summary.keys()))

        tensor_names = {k: v.name for k, v in summary.items()}
        tensor_names_must = {k: "/".join([model_summary.name, k]) + ':0'
                             for k in summary}
        self.assertDictEqual(tensor_names_must,
                             tensor_names)
