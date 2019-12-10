# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

from unittest.mock import MagicMock
from unittest.mock import PropertyMock
from unittest.mock import patch

from absl.testing import parameterized
import tensorflow as tf

from nucleus7.coordinator.configs import RunIterationInfo
from nucleus7.kpi.accumulator import KPIAccumulator
from nucleus7.kpi.kpi_callback import convert_evaluator_to_callback
from nucleus7.kpi.kpi_evaluator import KPIEvaluator
from nucleus7.kpi.kpi_plugin import KPIPlugin
from nucleus7.kpi.saver import TfSummaryKPISaver
from nucleus7.test_utils.test_utils import TestCaseWithReset


class TestKPICallback(TestCaseWithReset, parameterized.TestCase,
                      tf.test.TestCase):

    def setUp(self):
        super(TestKPICallback, self).setUp()
        self.data_batch = {
            "plugin1": "data1",
            "nucleutde2": "data2"}

    @parameterized.parameters(
        {"with_summary_writer": False, "epoch_number": 1,
         "iteration_number": 10, "is_last_iteration": False},
        {"with_summary_writer": True, "epoch_number": 1,
         "iteration_number": 10, "is_last_iteration": False},
        {"with_summary_writer": True, "epoch_number": 3,
         "iteration_number": 1, "is_last_iteration": True},
        {"with_summary_writer": True, "epoch_number": 3,
         "iteration_number": 10, "is_last_iteration": True},
    )
    @patch("nucleus7.kpi.kpi_evaluator.KPIEvaluator.__call__")
    @patch("nucleus7.kpi.kpi_evaluator.KPIEvaluator.is_last_iteration",
           new_callable=PropertyMock)
    def test_on_iteration_end(self, evaluator_is_last_iteration,
                              evaluator_call,
                              with_summary_writer, epoch_number,
                              iteration_number, is_last_iteration):
        def _evaluator_call(**inputs):
            return {"kpi": 1}

        log_dir = self.get_temp_dir()

        evaluator_is_last_iteration.side_effect = lambda x: x
        evaluator_call.side_effect = _evaluator_call
        accumulator = KPIAccumulator()
        plugin = KPIPlugin()

        evaluator = KPIEvaluator(plugin, accumulator).build()
        evaluator.clear_state = MagicMock(return_value=None)

        callback = convert_evaluator_to_callback(evaluator, 3)

        callback.iteration_info = RunIterationInfo(
            epoch_number, iteration_number, 0, is_last_iteration)

        if with_summary_writer:
            summary_writer = tf.summary.FileWriter(log_dir)
            callback.summary_writer = summary_writer
            callback.summary_step = 7

        callback.on_iteration_end(**self.data_batch)

        if epoch_number == 0 or epoch_number % 3 or iteration_number > 1:
            evaluator.clear_state.assert_not_called()
        else:
            evaluator.clear_state.assert_called_once_with()

        evaluator_call.assert_called_once_with(
            **self.data_batch)

        self.assertEmpty(plugin.savers)
        evaluator_is_last_iteration.assert_called_once_with(is_last_iteration)
        if not with_summary_writer:
            self.assertEmpty(accumulator.savers)
        else:
            self.assertEqual(1,
                             len(accumulator.savers))
            summary_saver = accumulator.savers[0]
            self.assertIsInstance(summary_saver,
                                  TfSummaryKPISaver)
            self.assertIs(callback.summary_writer,
                          summary_saver.summary_writer)
            self.assertEqual(7,
                             summary_saver.summary_step)
