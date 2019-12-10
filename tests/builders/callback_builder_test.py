# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

import tensorflow as tf

import nucleus7 as nc7
from nucleus7.builders import callback_builder
from nucleus7.test_utils.test_utils import register_new_class
from nucleus7.test_utils.test_utils import reset_register_and_logger


class TestCallbackBuilder(tf.test.TestCase):

    def setUp(self):
        super(TestCallbackBuilder, self).setUp()
        reset_register_and_logger()

    def test_get_callbacks(self):
        register_new_class('dummy_callback_1',
                           nc7.coordinator.CoordinatorCallback)
        register_new_class('dummy_callback_2',
                           nc7.coordinator.CoordinatorCallback)
        config_callbacks = [
            {'class_name': 'dummy_callback_1', 'name': 'callback1'},
            {'class_name': 'dummy_callback_2', 'name': 'callback2'}, ]
        callbacks = callback_builder.build_callbacks_chain(config_callbacks)
        for each_callback in callbacks:
            self.assertTrue(each_callback.built)
        self.assertEqual(2, len(callbacks))
        for callback in callbacks:
            self.assertIsInstance(callback, nc7.coordinator.CoordinatorCallback)
            self.assertTrue(callback.built)
