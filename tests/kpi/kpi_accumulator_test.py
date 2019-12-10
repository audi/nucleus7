# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

import os
from unittest.mock import MagicMock
from unittest.mock import call as mock_call

import numpy as np
import tensorflow as tf

from nucleus7.kpi.cacher import KPIMD5Cacher
from nucleus7.kpi.saver import KPIJsonSaver
from nucleus7.test_utils.model_dummies import DummyF1KPIAccumulator


class TestKPIAccumulator(tf.test.TestCase):

    def setUp(self):
        self.inputs = [
            {"prefix": "sample1", "true_positives": 0, "false_positives": 0,
             "false_negatives": 1},
            {"prefix": "sample2", "true_positives": 1, "false_positives": 0,
             "false_negatives": 0},
            {"prefix": "sample3", "true_positives": 0, "false_positives": 1,
             "false_negatives": 0},
            {"prefix": "sample4", "true_positives": 1, "false_positives": 0,
             "false_negatives": 0},
            {"prefix": "sample5", "true_positives": 0, "false_positives": 0,
             "false_negatives": 0},
            {"prefix": "sample6", "true_positives": 1, "false_positives": 0,
             "false_negatives": 0},
            {"prefix": "sample7", "true_positives": 0, "false_positives": 0,
             "false_negatives": 0},
        ]
        self.evaluate_flag = [0, 0, 0, 1, 0, 0, 0]
        self.kpis_must = [
            {"precision": None, "recall": None, "f1_score": None},
            {"precision": None, "recall": None, "f1_score": None},
            {"precision": None, "recall": None, "f1_score": None},
            self._get_kpi_must(0, 4),
            {"precision": None, "recall": None, "f1_score": None},
            {"precision": None, "recall": None, "f1_score": None},
            self._get_kpi_must(4, 7)
        ]

    def test_evaluate_on_sample(self):
        temp_dir = self.get_temp_dir()
        os.mkdir(os.path.join(temp_dir, "save"))
        os.mkdir(os.path.join(temp_dir, "cache"))

        saver = KPIJsonSaver(add_prefix_to_name=True).build()
        cacher = KPIMD5Cacher().build()
        kpi_accumulator = DummyF1KPIAccumulator(
            cachers=[cacher], savers=[saver]
        ).build()

        kpi_accumulator.save_target = os.path.join(temp_dir, "save")
        kpi_accumulator.cache_target = os.path.join(temp_dir, "cache")

        kpi_accumulator.process = MagicMock(wraps=kpi_accumulator.process)
        kpi_accumulator.buffer_processor.buffer.add = MagicMock(
            wraps=kpi_accumulator.buffer_processor.buffer.add)
        kpi_accumulator.clear_state = MagicMock(
            wraps=kpi_accumulator.clear_state)

        saver.save = MagicMock(wraps=saver.save)
        cacher.cache = MagicMock(wraps=cacher.cache)

        for i_sample, (each_evaluate_flag, each_inputs) in enumerate(
                zip(self.evaluate_flag, self.inputs)):
            is_last_sample = False
            if i_sample == len(self.inputs) - 1:
                is_last_sample = True
                kpi_accumulator.is_last_sample = True
            kpi = kpi_accumulator.evaluate_on_sample(
                evaluate=each_evaluate_flag,
                **each_inputs)
            if not each_evaluate_flag and not is_last_sample:
                self.assertIsNone(kpi)
            else:
                self.assertAllClose(self.kpis_must[i_sample],
                                    kpi)

        last_kpi_must = {k: v for k, v in self.kpis_must[-1].items()}
        self.assertAllClose(last_kpi_must,
                            kpi_accumulator.last_kpi)

        kpi_process_calls_must = [
            mock_call(true_positives=[0, 1, 0, 1],
                      false_positives=[0, 0, 1, 0],
                      false_negatives=[1, 0, 0, 0]),
            mock_call(true_positives=[0, 1, 0],
                      false_positives=[0, 0, 0],
                      false_negatives=[0, 0, 0]),
        ]
        kpi_accumulate_calls_must = [
            mock_call(**each_sample)
            for each_sample in self.inputs
        ]
        saver_save_calls_must = [
            mock_call(name="sample1-sample4-DummyF1KPIAccumulator",
                      values=self._get_kpi_must(0, 4)),
            mock_call(name="sample5-sample7-DummyF1KPIAccumulator",
                      values=self._get_kpi_must(4, 7)),
        ]
        cacher_cache_calls_must = [
            mock_call(self._get_kpi_must(0, 4)),
            mock_call(self._get_kpi_must(4, 7)),
        ]

        kpi_accumulator.process.assert_has_calls(kpi_process_calls_must)
        kpi_accumulator.buffer_processor.buffer.add.assert_has_calls(
            kpi_accumulate_calls_must)
        saver.save.assert_has_calls(saver_save_calls_must)
        cacher.cache.assert_has_calls(cacher_cache_calls_must)

        self.assertTrue(kpi_accumulator.buffer.is_empty())

    def _get_kpi_must(self, start_ind, last_ind):
        inputs_selected = self.inputs[start_ind: last_ind]
        tp = np.sum([each_sample["true_positives"]
                     for each_sample in inputs_selected])
        fp = np.sum([each_sample["false_positives"]
                     for each_sample in inputs_selected])
        fn = np.sum([each_sample["false_negatives"]
                     for each_sample in inputs_selected])

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1_score = 2 * precision * recall / (precision + recall)
        return {"precision": precision, "recall": recall, "f1_score": f1_score}
