# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

from functools import partial
from unittest.mock import MagicMock
from unittest.mock import PropertyMock
from unittest.mock import patch

from absl.testing import parameterized
import tensorflow as tf

from nucleus7.coordinator.callback import CoordinatorCallback
from nucleus7.coordinator.callbacks_handler import CallbacksHandler
from nucleus7.coordinator.configs import RunIterationInfo
from nucleus7.core.nucleotide import Nucleotide
from nucleus7.kpi.kpi_callback import convert_evaluator_to_callback
from nucleus7.kpi.kpi_evaluator import KPIEvaluator
from nucleus7.test_utils.test_utils import reset_register_and_logger


class TestCallbacksHandler(tf.test.TestCase, parameterized.TestCase):

    def setUp(self):
        reset_register_and_logger()

    @staticmethod
    def _get_callbacks_and_incoming_nucleotides():
        def side_effect_callback_on_iteration_end(
                nucleotide: Nucleotide, **inputs):
            return {k: '_'.join([nucleotide.name, k])
                    for k in nucleotide.generated_keys_all}

        input_node1 = Nucleotide(name='input_node1')
        input_node1.generated_keys = ['output1', 'output2']
        input_node2 = Nucleotide(name='input_node2')
        input_node2.generated_keys = ['output1']
        input_node1.process = MagicMock(return_value=None)
        input_node2.process = MagicMock(return_value=None)

        callback1 = CoordinatorCallback(
            name='callback1',
            inbound_nodes=['input_node1'],
            incoming_keys_mapping={
                'input_node1': {
                    'output1': 'input11'
                }
            })
        callback1.incoming_keys = ['input11']
        callback1.generated_keys = ['output11', 'output12']
        callback2 = CoordinatorCallback(
            name='callback2',
            inbound_nodes=['callback1', 'input_node2'],
            incoming_keys_mapping={
                'callback1': {
                    'output11': 'input22',
                },
                'input_node2': {
                    'output1': 'input21',
                }
            })
        callback2.incoming_keys = ['input21', '_input22']
        callback2.generated_keys = ['output21', 'output22']
        callback3 = CoordinatorCallback(
            name='callback3',
            inbound_nodes=['callback1', 'callback2'],
            incoming_keys_mapping={
                'callback1': {
                    'output11': 'input31',
                },
                'callback2': {
                    'output22': 'input32',
                }
            })
        callback3.incoming_keys = ['input31', 'input32']
        callback3.generated_keys = ['output31']

        callback1.on_iteration_end = MagicMock(side_effect=partial(
            side_effect_callback_on_iteration_end, nucleotide=callback1))
        callback2.on_iteration_end = MagicMock(side_effect=partial(
            side_effect_callback_on_iteration_end, nucleotide=callback2))
        callback3.on_iteration_end = MagicMock(side_effect=partial(
            side_effect_callback_on_iteration_end, nucleotide=callback3))

        callbacks = [callback1.build(), callback2.build(), callback3.build()]
        incoming_nucleotides = {'input_node1': input_node1.build(),
                                'input_node2': input_node2.build()}

        return callbacks, incoming_nucleotides

    def test_iteration_info_setter(self):
        iteration_info = RunIterationInfo(1, 100, 10)
        callbacks, _ = self._get_callbacks_and_incoming_nucleotides()
        callbacks_handler = CallbacksHandler(callbacks=callbacks).build()
        callbacks_handler.iteration_info = iteration_info
        for callback in callbacks:
            self.assertTupleEqual(tuple(iteration_info),
                                  tuple(callback.iteration_info))

    def test_number_iterations_per_epoch_setter(self):
        callbacks, _ = self._get_callbacks_and_incoming_nucleotides()
        callbacks_handler = CallbacksHandler(callbacks=callbacks).build()
        callbacks_handler.number_iterations_per_epoch = 123
        for callback in callbacks:
            self.assertEqual(123, callback.number_iterations_per_epoch)

    def test_log_dir_setter(self):
        callbacks, _ = self._get_callbacks_and_incoming_nucleotides()
        callbacks_handler = CallbacksHandler(callbacks=callbacks).build()
        callbacks_handler.log_dir = "current/log_dir"
        for callback in callbacks:
            self.assertEqual("current/log_dir", callback.log_dir)

    def test_summary_writer_setter(self):
        callbacks, _ = self._get_callbacks_and_incoming_nucleotides()
        callbacks_handler = CallbacksHandler(callbacks=callbacks).build()
        summary_writer = tf.summary.FileWriter(self.get_temp_dir())
        callbacks_handler.summary_writer = summary_writer
        for callback in callbacks:
            self.assertIs(summary_writer, callback.summary_writer)

    def test_summary_step_setter(self):
        callbacks, _ = self._get_callbacks_and_incoming_nucleotides()
        callbacks_handler = CallbacksHandler(callbacks=callbacks).build()
        callbacks_handler.summary_step = 120
        for callback in callbacks:
            self.assertEqual(120, callback.summary_step)

    def test_mode_step_setter(self):
        callbacks, _ = self._get_callbacks_and_incoming_nucleotides()
        callbacks_handler = CallbacksHandler(callbacks=callbacks).build()
        callbacks_handler.mode = 'TEST_MODE'
        for callback in callbacks:
            self.assertEqual('TEST_MODE', callback.mode)

    def test_call(self):
        (callbacks, incoming_nucleotides
         ) = self._get_callbacks_and_incoming_nucleotides()
        callbacks_handler = CallbacksHandler(callbacks=callbacks).build()
        callbacks_handler.build_dna(incoming_nucleotides)
        data = {
            'input_node1': {'output1': 0,
                            'output2': 1},
            'input_node2': {'output1': 3}
        }
        result = callbacks_handler(**data)
        result_must = {
            'callback1': {'output11': 'callback1_output11',
                          'output12': 'callback1_output12'},
            'callback2': {'output21': 'callback2_output21',
                          'output22': 'callback2_output22'},
            'callback3': {'output31': 'callback3_output31'}
        }
        self.assertDictEqual(result_must, result)

        callback1 = callbacks[0]
        callback2 = callbacks[1]
        callback3 = callbacks[2]

        callback1.on_iteration_end.assert_called_once_with(input11=0)
        callback2.on_iteration_end.assert_called_once_with(
            input21=3, input22='callback1_output11')
        callback3.on_iteration_end.assert_called_once_with(
            input31="callback1_output11", input32="callback2_output22")

    def test_evaluator_callbacks_getter(self):
        (callbacks, incoming_nucleotides
         ) = self._get_callbacks_and_incoming_nucleotides()
        evaluator = KPIEvaluator([], []).build()
        kpi_callback = convert_evaluator_to_callback(evaluator)
        callbacks.append(kpi_callback)
        callbacks_handler = CallbacksHandler(callbacks=callbacks).build()
        kpi_callbacks = callbacks_handler.kpi_evaluators
        self.assertListEqual([evaluator],
                             kpi_callbacks)

    @parameterized.parameters({"with_evaluator": True},
                              {"with_evaluator": False})
    @patch("nucleus7.kpi.kpi_evaluator.KPIEvaluator.dna_helix",
           new_callable=PropertyMock)
    def test_kpi_evaluators_dna_helices(self, evaluator_dna_helix,
                                        with_evaluator):
        evaluator_dna_helix.return_value = "evaluator_DNA_helix"
        (callbacks, incoming_nucleotides
         ) = self._get_callbacks_and_incoming_nucleotides()
        if with_evaluator:
            evaluator = KPIEvaluator([], []).build()
            kpi_callback = convert_evaluator_to_callback(evaluator)
            callbacks.append(kpi_callback)
        callbacks_handler = CallbacksHandler(callbacks=callbacks).build()
        kpi_dna_helices = callbacks_handler.kpi_evaluators_dna_helices
        if not with_evaluator:
            self.assertIsNone(kpi_dna_helices)
        else:
            self.assertDictEqual({"kpi_evaluator": "evaluator_DNA_helix"},
                                 kpi_dna_helices)

    def test_begin_call_every_callback(self):
        callbacks, _ = self._get_callbacks_and_incoming_nucleotides()
        callbacks_handler = CallbacksHandler(callbacks=callbacks).build()
        for callback in callbacks:
            callback.begin = MagicMock(return_value=None)
        callbacks_handler.begin()
        for callback in callbacks:
            self.assertEqual(1,
                             callback.begin.call_count)

    def test_on_iteration_start_call_every_callback(self):
        callbacks, _ = self._get_callbacks_and_incoming_nucleotides()
        callbacks_handler = CallbacksHandler(callbacks=callbacks).build()
        for callback in callbacks:
            callback.on_iteration_start = MagicMock(return_value=None)
        callbacks_handler.on_iteration_start()
        for callback in callbacks:
            self.assertEqual(1,
                             callback.on_iteration_start.call_count)

    def test_end_call_every_callback(self):
        callbacks, _ = self._get_callbacks_and_incoming_nucleotides()
        callbacks_handler = CallbacksHandler(callbacks=callbacks).build()
        for callback in callbacks:
            callback.end = MagicMock(return_value=None)
        callbacks_handler.end()
        for callback in callbacks:
            self.assertEqual(1,
                             callback.end.call_count)
