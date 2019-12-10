# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

import os

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from nucleus7.core.nucleotide import Nucleotide
from nucleus7.kpi.accumulator import KPIAccumulator
from nucleus7.kpi.cacher import KPIMD5Cacher
from nucleus7.kpi.kpi_evaluator import KPIEvaluator
from nucleus7.kpi.saver import KPIJsonSaver
from nucleus7.test_utils.model_dummies import DummyF1KPIAccumulator
from nucleus7.test_utils.model_dummies import DummyTpFpTnFnKPIPlugin
from nucleus7.test_utils.test_utils import TestCaseWithReset
from nucleus7.utils import nest_utils


class _MeanKPIAccumulator(KPIAccumulator):
    exclude_from_register = True
    incoming_keys = [
        "f1_score",
        "precision",
        "recall",
    ]
    generated_keys = [
        "f1_score",
        "precision",
        "recall",
    ]

    def process(self, **inputs):
        return {each_key: np.mean(inputs[each_key])
                for each_key in self.incoming_keys}


class TestKPIEvaluator(TestCaseWithReset, tf.test.TestCase,
                       parameterized.TestCase):

    def setUp(self):
        self.num_samples = 7
        self.data = [
            {"prefix": "sample1", "labels": 1, "predictions": 0, "evaluate": 0},
            {"prefix": "sample2", "labels": 1, "predictions": 1, "evaluate": 0},
            {"prefix": "sample3", "labels": 0, "predictions": 1, "evaluate": 0},
            {"prefix": "sample4", "labels": 1, "predictions": 1, "evaluate": 1},
            {"prefix": "sample5", "labels": 0, "predictions": 0, "evaluate": 0},
            {"prefix": "sample6", "labels": 0, "predictions": 0, "evaluate": 0},
            {"prefix": "sample7", "labels": 1, "predictions": 1, "evaluate": 0},
        ]

        kpi1_must = [self._get_kpi_must(0, 4), self._get_kpi_must(4, 7)]
        kpi2_must = {k: (kpi1_must[0][k] + kpi1_must[1][k]) / 2
                     for k in kpi1_must[0]}

        self.kpis1_must = [
            {},
            {},
            {},
            kpi1_must[0],
            {},
            {},
            kpi1_must[1],
        ]
        self.kpis2_must = [
            {},
            {},
            {},
            {},
            {},
            {},
            kpi2_must,
        ]

    @parameterized.parameters({"is_last_iteration": True},
                              {"is_last_iteration": False})
    def test_kpi_call(self, is_last_iteration):
        temp_dir = self.get_temp_dir()
        os.mkdir(os.path.join(temp_dir, "save"))
        os.mkdir(os.path.join(temp_dir, "cache"))

        cacher1 = KPIMD5Cacher().build()
        kpi_plugin = DummyTpFpTnFnKPIPlugin(
            name="kpi_plugin1",
            inbound_nodes=["dataset"],
            cachers=[cacher1]).build()

        saver2 = KPIJsonSaver().build()
        cacher2 = KPIMD5Cacher().build()
        kpi_accumulator1 = DummyF1KPIAccumulator(
            name="kpi_acc1",
            inbound_nodes=["kpi_plugin1", "dataset"],
            cachers=[cacher2], savers=[saver2]).build()

        saver3 = KPIJsonSaver().build()
        cacher3 = KPIMD5Cacher().build()
        kpi_accumulator2 = _MeanKPIAccumulator(
            name="kpi_acc2",
            inbound_nodes=["kpi_acc1", "dataset"],
            incoming_keys_mapping={"dataset": {"evaluate": "_"}},
            cachers=[cacher3], savers=[saver3]).build()

        kpi_evaluator = KPIEvaluator(
            plugins=kpi_plugin,
            accumulators=[kpi_accumulator1, kpi_accumulator2]).build()
        kpi_evaluator.save_target = os.path.join(temp_dir, "save")
        kpi_evaluator.cache_target = os.path.join(temp_dir, "cache")
        dataset_nucleotide = Nucleotide(name="dataset").build()
        dataset_nucleotide.generated_keys = [
            "labels", "predictions", "evaluate", "prefix"]
        incoming_nucleotides = {'dataset': dataset_nucleotide}
        kpi_evaluator.build_dna(incoming_nucleotides)

        data_batch = nest_utils.combine_nested(self.data, np.array)
        kpi_evaluator.is_last_iteration = is_last_iteration
        _ = kpi_evaluator(dataset=data_batch)

        if is_last_iteration:
            last_kpi_must = {
                "kpi_acc1": self.kpis1_must[-1],
                "kpi_acc2": self.kpis2_must[-1]}
        else:
            last_kpi_must = {
                "kpi_acc1": self.kpis1_must[3]}
        self.assertAllClose(last_kpi_must,
                            kpi_evaluator.last_kpi)

    def _get_kpi_must(self, start_ind, last_ind):
        kpi_intermediate = [
            {"true_positives": 0, "false_positives": 0, "false_negatives": 1},
            {"true_positives": 1, "false_positives": 0, "false_negatives": 0},
            {"true_positives": 0, "false_positives": 1, "false_negatives": 0},
            {"true_positives": 1, "false_positives": 0, "false_negatives": 0},
            {"true_positives": 0, "false_positives": 0, "false_negatives": 0},
            {"true_positives": 0, "false_positives": 0, "false_negatives": 0},
            {"true_positives": 1, "false_positives": 0, "false_negatives": 0},
        ]

        inputs_selected = kpi_intermediate[start_ind: last_ind]
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
