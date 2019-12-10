# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

import json
import os
from unittest.mock import call as mock_call
from unittest.mock import patch

import numpy as np
import tensorflow as tf

from nucleus7.kpi.saver import KPIJsonSaver
from nucleus7.kpi.saver import MlflowKPILogger


class TestKPIJsonSaver(tf.test.TestCase):
    def setUp(self):
        self._kpi_values = {
            "a": 100,
            "b": np.random.rand(2),
            "c": {"c1": 5,
                  "c2": np.random.rand(10, 5)},
            "d": [1, 2, 3]
        }
        self._kpi_values_saved_must = {
            "a": 100,
            "c": {"c1": 5},
            "d": [1, 2, 3]
        }

    def test_save(self):
        saver = KPIJsonSaver().build()
        saver.save_target = self.get_temp_dir()
        saver.save("kpi1", self._kpi_values)

        with open(os.path.join(self.get_temp_dir(), "kpi1.json"), "r") as f:
            values_restored = json.load(f)

        self.assertAllClose(self._kpi_values_saved_must,
                            values_restored)


class TestMlflowKPILogger(tf.test.TestCase):
    def setUp(self):
        self._kpi_values = {
            "a": 100,
            "b": np.random.rand(2),
            "c": {"c1": 5,
                  "c2": np.random.rand(10, 5)},
            "d": [1, 2, 3]
        }
        self._kpi_values_logged_must = {
            "a": 100,
            "c--c1": 5,
            "d--0": 1,
            "d--1": 2,
            "d--2": 3,
        }

    @patch("nucleus7.utils.mlflow_utils.log_metric_to_mlflow")
    def test_save(self, mlflow_log_metric_fn):
        def _log_metric(param_name, param_value):
            return

        mlflow_log_metric_fn.side_effect = _log_metric

        saver = MlflowKPILogger().build()
        saver.save("kpi1", self._kpi_values)

        calls_must = [
            mock_call("_".join(["kpi1", each_key]), each_value)
            for each_key, each_value in sorted(
                self._kpi_values_logged_must.items())]
        mlflow_log_metric_fn.assert_has_calls(calls_must)
