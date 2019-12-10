# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

from collections import namedtuple
import copy
import json
import os
from unittest.mock import call as mock_call
from unittest.mock import patch

from nucleus7.core import project_serializer
from nucleus7.test_utils.test_utils import TestCaseWithTempDir
from nucleus7.test_utils.test_utils import reset_register_and_logger
from nucleus7.utils import nest_utils

_Config = namedtuple("Config", ["nt_p1", "nt_p2"])


class _NotSerializableInstance(object):
    def __init__(self):
        self.a = 10


class TestRunConfigSerializer(TestCaseWithTempDir):

    def setUp(self):
        self.maxDiff = None
        super(TestRunConfigSerializer, self).setUp()
        reset_register_and_logger()
        self.configs_to_log, self.config_serialized_must = _get_configs_to_log()
        self.configs_to_log_copy = copy.copy(self.configs_to_log)

    @patch("nucleus7.core.config_logger.get_logged_configs")
    def test_serialize(self, get_logged_configs_fn):
        get_logged_configs_fn.return_value = self.configs_to_log
        save_dir = self.get_temp_dir()
        run_name = "temp_run"
        single_config_names = ["model", "dataset"]
        serializer = project_serializer.RunConfigSerializer(
            save_dir=save_dir, name=run_name,
            single_config_names=single_config_names
        )
        serializer.serialize()
        fname_must = os.path.join(save_dir, "_config_{}.json".format(run_name))
        self.assertTrue(os.path.isfile(fname_must))
        with open(fname_must, "r") as f:
            config_load = json.load(f)
        self.assertDictEqual(self.config_serialized_must,
                             config_load)
        self.assertDictEqual(self.configs_to_log_copy,
                             self.configs_to_log)


class TestMlflowConfigSerializer(TestCaseWithTempDir):

    def setUp(self):
        super(TestMlflowConfigSerializer, self).setUp()
        self.configs_to_log, self.config_serialized_must = (
            _get_configs_to_log(False))
        self.configs_to_log_copy = copy.copy(self.configs_to_log)

    @patch("nucleus7.utils.mlflow_utils.log_config_parameter_to_mlflow")
    def test_serialize_to_file(self, log_config_parameter_to_mlflow_fn):
        def _log_config_parameter_to_mlflow(param_name, param_value):
            pass

        log_config_parameter_to_mlflow_fn.side_effect = (
            _log_config_parameter_to_mlflow)

        single_config_names = ["model", "dataset"]
        serializer = project_serializer.MlflowConfigSerializer(
            save_dir="",
            single_config_names=single_config_names
        )
        serializer.serialize_to_file(self.configs_to_log)
        config_serialized_must_flatten = nest_utils.flatten_nested_struct(
            self.config_serialized_must, separator="/")

        for each_param_name, each_param_value in (
                config_serialized_must_flatten.items()):
            log_config_parameter_to_mlflow_fn.assert_has_calls(
                [mock_call(each_param_name, each_param_value)])

        log_config_parameter_to_mlflow_fn.assert_has_calls(
            [mock_call("CLUSTER_SPEC", {})])
        len_of_calls_must = len(config_serialized_must_flatten) + 1
        self.assertEqual(len_of_calls_must,
                         log_config_parameter_to_mlflow_fn.call_count)

        self.assertDictEqual(self.configs_to_log_copy,
                             self.configs_to_log)


def _get_configs_to_log(not_serializable_as_text=True):
    not_serializable_instance = _NotSerializableInstance()
    configs_to_log = {
        "model": {"Model": [{"p1": 10, "p2": _Config(1, 2),
                             "numpy": not_serializable_instance}]},
        "plugin": {"Plugin1": [{"p1": 20}],
                   "Plugin2": [{"p2": 50}]},
        "dataset": {"Dataset": [
            {"_get_mode": lambda: "train", "p1": "train"},
            {"_get_mode": lambda: "eval", "p1": "eval"}]},
        "callback": {
            "Callback1": [
                {"p1": 10, "p2": -20, "_get_mode": lambda: "train"},
                {"p3": 30, "_get_mode": lambda: "eval"}],
            "Callback2": [{"p5": 50, "_get_mode": lambda: "eval"}]}
    }
    not_serializable_serialized = (
        "NOT_SERIALIZABLE" if not_serializable_as_text
        else not_serializable_instance)
    config_serialized_must = {
        "model_config": {"p1": 10, "p2": {"nt_p1": 1, "nt_p2": 2},
                         "numpy": not_serializable_serialized},
        "plugin_configs": [{"p1": 20}, {"p2": 50}],
        "dataset_config": {"train": {"p1": "train"},
                           "eval": {"p1": "eval"}},
        "callback_configs": {"train": [{"p1": 10, "p2": -20}],
                             "eval": [{"p3": 30}, {"p5": 50}]}
    }
    return configs_to_log, config_serialized_must
