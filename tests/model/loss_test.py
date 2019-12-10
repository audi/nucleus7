# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

from unittest.mock import MagicMock

import numpy as np
import tensorflow as tf

from nucleus7.test_utils.model_dummies import DummySoftmaxLoss
from nucleus7.test_utils.test_utils import reset_register_and_logger


class TestModelLoss(tf.test.TestCase):
    def setUp(self):
        tf.reset_default_graph()
        reset_register_and_logger()
        np.random.seed(5467)

    def test_call(self):
        model_loss = DummySoftmaxLoss(inbound_nodes=[], name='softmaxloss')
        model_loss.build()
        model_loss.mode = 'train'
        inputs = {'labels': tf.placeholder(tf.int64, shape=[None] * 3),
                  'logits': tf.placeholder(tf.float32, shape=[None] * 3 + [10])}
        losses = model_loss(**inputs)
        self.assertIsInstance(losses, dict)
        self.assertSetEqual({'loss', 'total_loss'}, set(losses.keys()))
        self.assertEqual(model_loss._variable_scope, 'softmaxloss')

        tensor_names = {k: v.name for k, v in losses.items()}
        tensor_names_must = {k: "/".join([model_loss.name, k]) + ':0'
                             for k in losses}
        self.assertDictEqual(tensor_names_must,
                             tensor_names)

    def test_call_with_loss_weights(self):
        loss_weights = {"loss1": 0.1,
                        "loss2": 0.5}
        loss_values = {"loss1": 1.0,
                       "loss2": 2.0,
                       "loss3": 3.0}
        inputs = {"labels": "labels",
                  "logits": "logits"}
        model_loss = DummySoftmaxLoss(inbound_nodes=[], name='softmaxloss',
                                      loss_weights=loss_weights)
        model_loss.process = MagicMock(return_value=loss_values)
        model_loss.build()
        model_loss.mode = "train"
        result = model_loss(**inputs)
        self.assertAllClose(4.1,
                            result["total_loss"])
