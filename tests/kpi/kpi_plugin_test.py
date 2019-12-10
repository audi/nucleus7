# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

import os
from unittest.mock import MagicMock
from unittest.mock import PropertyMock
from unittest.mock import call as mock_call
from unittest.mock import patch

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from nucleus7.kpi.cacher import KPIMD5Cacher
from nucleus7.kpi.saver import KPIJsonSaver
from nucleus7.test_utils.model_dummies import DummyTpFpTnFnKPIPlugin
from nucleus7.utils import nest_utils


class TestKPIPlugin(parameterized.TestCase, tf.test.TestCase):

    def setUp(self):
        self.num_samples = 6
        self.data = [
            {"prefix": "sample1", "labels": 1, "predictions": 0},
            {"prefix": "sample2", "labels": 1, "predictions": 1},
            {"prefix": "sample3", "labels": 0, "predictions": 1},
            {"prefix": "sample4", "labels": 1, "predictions": 1},
            {"prefix": "sample5", "labels": 0, "predictions": 0},
            {"prefix": "sample6", "labels": 0, "predictions": 0},
        ]
        self.kpi_must = [
            {"true_positives": 0, "false_positives": 0,
             "true_negatives": 0, "false_negatives": 1},
            {"true_positives": 1, "false_positives": 0,
             "true_negatives": 0, "false_negatives": 0},
            {"true_positives": 0, "false_positives": 1,
             "true_negatives": 0, "false_negatives": 0},
            {"true_positives": 1, "false_positives": 0,
             "true_negatives": 0, "false_negatives": 0},
            {"true_positives": 0, "false_positives": 0,
             "true_negatives": 1, "false_negatives": 0},
            {"true_positives": 0, "false_positives": 0,
             "true_negatives": 1, "false_negatives": 0},
        ]

    @parameterized.parameters({"add_prefix_to_saver_name": True},
                              {"add_prefix_to_saver_name": False})
    def test_evaluate_on_sample(self, add_prefix_to_saver_name):
        temp_dir = self.get_temp_dir()
        os.mkdir(os.path.join(temp_dir, "save"))
        os.mkdir(os.path.join(temp_dir, "cache"))

        saver = KPIJsonSaver().build()
        saver.add_prefix_to_name = add_prefix_to_saver_name
        cacher = KPIMD5Cacher().build()
        kpi_plugin = DummyTpFpTnFnKPIPlugin(
            cachers=[cacher], savers=[saver]
        ).build()

        kpi_plugin.save_target = os.path.join(temp_dir, "save")
        kpi_plugin.cache_target = os.path.join(temp_dir, "cache")

        self.assertEqual(saver.save_target,
                         os.path.join(temp_dir, "save"))
        self.assertEqual(cacher.cache_target,
                         os.path.join(temp_dir, "cache"))
        kpi_plugin.process = MagicMock(wraps=kpi_plugin.process)
        saver.save = MagicMock(wraps=saver.save)
        cacher.cache = MagicMock(wraps=cacher.cache)
        cacher.restore = MagicMock(wraps=cacher.restore)
        cacher.calculate_hash = MagicMock(wraps=cacher.calculate_hash)

        for i_sample, each_sample in enumerate(self.data):
            kpi = kpi_plugin.evaluate_on_sample(**each_sample)
            self.assertAllClose(self.kpi_must[i_sample],
                                kpi)

        plugin_process_calls_must = [
            mock_call(labels=each_sample["labels"],
                      predictions=each_sample["predictions"])
            for each_sample in self.data
        ]
        saver_save_calls_must = [
            mock_call(
                name=(each_sample["prefix"] + "-" + "DummyTpFpTnFnKPIPlugin"
                      if add_prefix_to_saver_name
                      else "DummyTpFpTnFnKPIPlugin"),
                values=each_kpi)
            for each_sample, each_kpi in zip(self.data, self.kpi_must)
        ]
        cacher_cache_calls_must = [
            mock_call(each_kpi) for each_kpi in self.kpi_must
        ]
        cacher_calculate_hash_calls_must = [
            mock_call(
                cache_prefix=(each_sample["prefix"] + "-" +
                              "DummyTpFpTnFnKPIPlugin"),
                inputs={k: v for k, v in each_sample.items() if k != "prefix"})
            for each_sample in self.data
        ]

        kpi_plugin.process.assert_has_calls(plugin_process_calls_must)
        saver.save.assert_has_calls(saver_save_calls_must)
        cacher.cache.assert_has_calls(cacher_cache_calls_must)
        cacher.calculate_hash.assert_has_calls(
            cacher_calculate_hash_calls_must)
        self.assertEqual(len(self.data),
                         cacher.restore.call_count)

        # create new plugin and calculate KPI again but from cache
        saver2 = KPIJsonSaver().build()
        saver2.add_prefix_to_name = add_prefix_to_saver_name
        cacher2 = KPIMD5Cacher().build()
        kpi_plugin2 = DummyTpFpTnFnKPIPlugin(
            cachers=[cacher2], savers=[saver2]
        ).build()
        kpi_plugin2.save_target = os.path.join(temp_dir, "save")
        kpi_plugin2.cache_target = os.path.join(temp_dir, "cache")
        kpi_plugin2.process = MagicMock(wraps=kpi_plugin2.process)
        saver2.save = MagicMock(wraps=saver2.save)
        cacher2.cache = MagicMock(wraps=cacher2.cache)
        cacher2.restore = MagicMock(wraps=cacher2.restore)

        for i_sample, each_sample in enumerate(self.data):
            kpi2 = kpi_plugin2.evaluate_on_sample(**each_sample)
            self.assertAllClose(self.kpi_must[i_sample],
                                kpi2)
        kpi_plugin2.process.assert_not_called()
        saver2.save.assert_has_calls(saver_save_calls_must)
        cacher2.cache.assert_not_called()
        self.assertEqual(len(self.data),
                         cacher2.restore.call_count)

    @parameterized.parameters(
        {"sample_mask": None, "is_last_iteration": False,
         "is_last_sample_must": [0, 0, 0, 0, 0, 0]},
        {"sample_mask": None, "is_last_iteration": True,
         "is_last_sample_must": [0, 0, 0, 0, 0, 1]},
        {"sample_mask": [1, 1, 1, 1, 1, 1], "is_last_iteration": False,
         "is_last_sample_must": [0, 0, 0, 0, 0, 0]},
        {"sample_mask": [0, 0, 0, 0, 0, 0], "is_last_iteration": False,
         "is_last_sample_must": []},
        {"sample_mask": [0, 0, 1, 1, 0, 0], "is_last_iteration": False,
         "is_last_sample_must": [0, 0]},
        {"sample_mask": [1, 1, 1, 1, 1, 1], "is_last_iteration": True,
         "is_last_sample_must": [0, 0, 0, 0, 0, 1]},
        {"sample_mask": [0, 0, 0, 0, 0, 0], "is_last_iteration": True,
         "is_last_sample_must": []},
        {"sample_mask": [0, 0, 1, 1, 0, 0], "is_last_iteration": True,
         "is_last_sample_must": [0, 1]},
    )
    @patch("nucleus7.kpi.kpi_plugin.KPIPlugin.is_last_sample",
           new_callable=PropertyMock)
    def test_call(self, plugin_is_last_sample, sample_mask, is_last_iteration,
                  is_last_sample_must):
        temp_dir = self.get_temp_dir()
        os.mkdir(os.path.join(temp_dir, "save"))
        os.mkdir(os.path.join(temp_dir, "cache"))

        plugin_is_last_sample.side_effect = lambda x: x
        saver = KPIJsonSaver().build()
        cacher = KPIMD5Cacher().build()
        kpi_plugin = DummyTpFpTnFnKPIPlugin(
            cachers=[cacher], savers=[saver]
        ).build()

        kpi_plugin.save_target = os.path.join(temp_dir, "save")
        kpi_plugin.cache_target = os.path.join(temp_dir, "cache")
        kpi_plugin.evaluate_on_sample = MagicMock(
            wraps=kpi_plugin.evaluate_on_sample)

        data_batch = nest_utils.combine_nested(self.data, combine_fun=np.stack)
        kpi_plugin.is_last_iteration = is_last_iteration

        kpi_must_list = []
        for i_sample, each_kpi_must in enumerate(self.kpi_must):
            if sample_mask is None or sample_mask[i_sample]:
                kpi_must_list.append(each_kpi_must)

        if kpi_must_list:
            kpi_must = nest_utils.combine_nested(kpi_must_list,
                                                 combine_fun=np.array)
        else:
            kpi_must = None

        is_last_sample_calls_must = [mock_call(i) for i in is_last_sample_must]
        if sample_mask is None:
            evaluate_on_sample_args_must = [
                mock_call(**each_sample_data) for each_sample_data in self.data]
        else:
            evaluate_on_sample_args_must = [
                mock_call(**each_sample_data)
                for i, each_sample_data in enumerate(self.data)
                if sample_mask[i]]

        kpi = kpi_plugin(sample_mask=sample_mask, **data_batch)

        plugin_is_last_sample.assert_has_calls(is_last_sample_calls_must)
        kpi_plugin.evaluate_on_sample.assert_has_calls(
            evaluate_on_sample_args_must)

        if kpi_must is None:
            self.assertIsNone(kpi)
            return

        if sample_mask is None:
            self.assertAllClose(kpi_must,
                                kpi)
        else:
            for i in range(sum(sample_mask)):
                sample_kpi_must = {k: v[i] for k, v in kpi_must.items()}
                sample_kpi = {k: v[i] for k, v in kpi.items()}
                if sample_mask[i]:
                    self.assertAllClose(sample_kpi_must,
                                        sample_kpi)
                else:
                    self.assertAllEqual(sample_kpi_must,
                                        sample_kpi)
