# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

import os
from unittest.mock import MagicMock
from unittest.mock import Mock
from unittest.mock import call as mock_call

from absl.testing import parameterized
import tensorflow as tf

from nucleus7.coordinator.callback import CoordinatorCallback
from nucleus7.coordinator.callbacks_handler import CallbacksHandler
from nucleus7.coordinator.session_run_hooks import (
    CustomNucleotideInitializerHook)
from nucleus7.coordinator.session_run_hooks import (
    convert_callback_to_session_hook)
from nucleus7.coordinator.session_run_hooks import (
    convert_callbacks_handler_to_session_hook)
from nucleus7.model.fields import CollectionNames
from nucleus7.utils import tf_collections_utils


class TestSessionRunHooks(parameterized.TestCase, tf.test.TestCase):

    def setUp(self):
        self.log_dir_main = self.get_temp_dir()
        self.summary_dir_main = self.get_temp_dir()
        self.iteration_nums = {'train': 3, 'eval': 1}
        self.epoch_num = 3
        self.iters_per_epoch = {'train': 10, 'eval': 8}
        self.max_iters_per_epoch = max(self.iters_per_epoch.values())

    @parameterized.parameters({'mode': 'eval'},
                              {'mode': 'train'})
    def test_convert_callback_to_session_hook(self, mode='train'):
        summary_step = self._get_summary_step(mode)
        global_step = self._get_global_step_value(mode)
        summary_dir_must = self._get_summary_dir_for_mode(mode)

        def mock_callback_iteration(image, inputs, outputs):
            pass

        callback = CoordinatorCallback(
            inbound_nodes=['node1', 'node2'],
            incoming_keys_mapping={'node1': {'predictions': 'image'}})
        callback.incoming_keys = ['image', 'inputs', 'outputs']
        callback.on_iteration_end = MagicMock(
            side_effect=mock_callback_iteration)
        callback.build()
        callback.begin = MagicMock(return_value=None)
        callback.on_iteration_start = MagicMock(return_value=None)
        callback.end = MagicMock(return_value=None)

        callback_calls_mock = Mock()
        callback_calls_mock.attach_mock(callback.begin,
                                        'begin_call')
        callback_calls_mock.attach_mock(callback.on_iteration_start,
                                        'on_iteration_start_call')
        callback_calls_mock.attach_mock(callback.on_iteration_end,
                                        'on_iteration_end_call')
        callback_calls_mock.attach_mock(callback.end,
                                        'end_call')

        tf.reset_default_graph()

        data = {'node1': {'predictions': tf.constant(1),
                          'inputs': tf.constant(2)},
                'node2': {'outputs': tf.constant(3)},
                'node3': {'outputs': tf.constant(4)}}

        global_step = tf.Variable(global_step, trainable=False,
                                  collections=[tf.GraphKeys.GLOBAL_VARIABLES,
                                               tf.GraphKeys.GLOBAL_STEP])
        summary_writer_must = tf.summary.FileWriterCache.get(summary_dir_must)
        callback.log_dir = self.log_dir_main
        callback.mode = mode
        callback.number_iterations_per_epoch = self.iters_per_epoch[mode]
        callback_hook = convert_callback_to_session_hook(
            callback, self.summary_dir_main,
            self.max_iters_per_epoch)
        self.assertIsInstance(callback_hook, tf.train.SessionRunHook)

        tf_collections_utils.nested2collection(
            CollectionNames.PREDICTIONS, data)

        args2callback_must = {
            'image': 1,
            'inputs': 2,
            'outputs': 3,
        }

        callbacks_expected_calls = [
            mock_call.begin_call(),
            mock_call.on_iteration_start_call(),
            mock_call.on_iteration_end_call(**args2callback_must),
            mock_call.end_call()
        ]

        hooks = [callback_hook]
        with tf.train.MonitoredTrainingSession(hooks=hooks) as mon_sess:
            mon_sess.run(data)

        callback_calls_mock.assert_has_calls(callbacks_expected_calls)
        self.assertEqual(callback.iteration_info.iteration_number,
                         self.iteration_nums[mode])
        self.assertEqual(callback.iteration_info.epoch_number,
                         self.epoch_num)
        self.assertIsInstance(callback.iteration_info.session_run_context,
                              tf.train.SessionRunContext)
        self.assertEqual(callback.summary_step, summary_step)
        self.assertIs(callback.summary_writer, summary_writer_must)

    @parameterized.parameters({'mode': 'eval'},
                              {'mode': 'train'})
    def test_convert_callbacks_handler_to_session_hook(self, mode='train'):
        class CallbacksHandlerMock(CallbacksHandler):
            def __init__(self, **kwargs):
                super(CallbacksHandlerMock, self).__init__(**kwargs)
                self._inbound_nodes = None

            @property
            def inbound_nodes(self):
                return self._inbound_nodes

        def _side_effect_callbacks_handler_process_gene(gene_name, gene_inputs):
            pass

        tf.reset_default_graph()

        summary_step = self._get_summary_step(mode)
        global_step = self._get_global_step_value(mode)
        summary_dir_must = self._get_summary_dir_for_mode(mode)

        side_effect_callbacks_handler = (
            _side_effect_callbacks_handler_process_gene)

        data = {'node1': {'predictions': tf.constant(1),
                          'inputs': tf.constant(2)},
                'node2': {'outputs': tf.constant(3)},
                'node3': {'outputs': tf.constant(4)}}

        inputs_to_handler_must = {
            'node1': {'predictions': 1,
                      'inputs': 2},
            'node2': {'outputs': 3}
        }

        handler = CallbacksHandlerMock(callbacks=[]).build()
        handler._inbound_nodes = ['node1', 'node2']
        handler.mode = mode
        handler.process_gene = MagicMock(
            side_effect=side_effect_callbacks_handler)
        handler.begin = MagicMock(return_value=None)
        handler.on_iteration_start = MagicMock(return_value=None)
        handler.end = MagicMock(return_value=None)

        handler_calls_mock = Mock()
        handler_calls_mock.attach_mock(handler.begin,
                                       'begin_call')
        handler_calls_mock.attach_mock(handler.on_iteration_start,
                                       'on_iteration_start_call')
        handler_calls_mock.attach_mock(handler.process_gene,
                                       'process_gene_call')
        handler_calls_mock.attach_mock(handler.end,
                                       'end_call')

        global_step = tf.Variable(global_step, trainable=False,
                                  collections=[tf.GraphKeys.GLOBAL_VARIABLES,
                                               tf.GraphKeys.GLOBAL_STEP])
        summary_writer_must = tf.summary.FileWriterCache.get(summary_dir_must)
        handler.log_dir = self.log_dir_main
        handler.mode = mode
        handler.number_iterations_per_epoch = (
            self.iters_per_epoch[mode])
        handler_hook = convert_callbacks_handler_to_session_hook(
            handler, self.summary_dir_main,
            self.max_iters_per_epoch
        )

        self.assertIsInstance(handler_hook, tf.train.SessionRunHook)
        self.assertTrue(os.path.isdir(os.path.join(summary_dir_must)))

        hooks = [handler_hook]
        tf_collections_utils.nested2collection(
            CollectionNames.PREDICTIONS, data)

        handler_expected_calls = [
            mock_call.begin_call(),
            mock_call.on_iteration_start_call(),
            mock_call.process_gene_call(gene_name="callbacks",
                                        gene_inputs=inputs_to_handler_must),
            mock_call.end_call()
        ]

        with tf.train.MonitoredTrainingSession(hooks=hooks) as mon_sess:
            mon_sess.run(data)

        handler_calls_mock.assert_has_calls(handler_expected_calls)

        self.assertEqual(handler.iteration_info.iteration_number,
                         self.iteration_nums[mode])
        self.assertEqual(handler.iteration_info.epoch_number,
                         self.epoch_num)
        self.assertEqual(handler.summary_step, summary_step)
        self.assertIs(handler.summary_writer, summary_writer_must)

    def _get_summary_step(self, mode):
        summary_step = (
            self.max_iters_per_epoch * self.epoch_num -
            self.iters_per_epoch[mode] + self.iteration_nums[mode]
        )
        return summary_step

    def _get_global_step_value(self, mode):
        if mode == 'train':
            global_step = ((self.epoch_num - 1) * self.iters_per_epoch['train']
                           + self.iteration_nums['train'] - 1)
        else:
            global_step = self.epoch_num * self.iters_per_epoch['train']
        return global_step

    def _get_summary_dir_for_mode(self, mode):
        return os.path.join(self.summary_dir_main, mode)


class TestCustomNucleotideInitializerHook(tf.test.TestCase):
    def test_after_create_session(self):
        class _Nucleotide:
            def __init__(self_, name):
                self_.name = name

        nucleotides_with_init = [_Nucleotide("nucleotide2"),
                                 _Nucleotide("nucleotide3")]

        for each_nucleotide in nucleotides_with_init:
            each_nucleotide.initialize_session = MagicMock(return_value=None)

        nucleotides = ([_Nucleotide("nucleotide1")]
                       + nucleotides_with_init
                       + [_Nucleotide("nucleotide4")])

        hook = CustomNucleotideInitializerHook(nucleotides)
        hook.after_create_session(tf.Session(), None)

        for each_nucleotide in nucleotides_with_init:
            each_nucleotide.initialize_session.assert_called_once_with()
