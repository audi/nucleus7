# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

import json
import os
from unittest.mock import ANY
from unittest.mock import patch

from absl.testing import parameterized
import mlflow
from mlflow.entities import run as mlflow_run
from mlflow.entities import run_info as mlflow_run_info

from nucleus7.test_utils.test_utils import TestCaseWithReset
from nucleus7.test_utils.test_utils import TestCaseWithTempDir
from nucleus7.utils import mlflow_utils


class TestMlflowUtils(TestCaseWithReset, parameterized.TestCase,
                      TestCaseWithTempDir):
    def setUp(self):
        TestCaseWithReset.setUp(self)
        TestCaseWithTempDir.setUp(self)
        if "MLFLOW_TRACKING_URI" in os.environ:
            del os.environ["MLFLOW_TRACKING_URI"]
        mlflow.set_tracking_uri(None)

    @parameterized.parameters({"with_uri": True, "name_from_meta": True},
                              {"with_uri": False, "name_from_meta": True},
                              {"with_uri": True, "name_from_meta": False},
                              {"with_uri": False, "name_from_meta": False})
    def test_create_new_or_continue_experiment(self, with_uri, name_from_meta):
        temp_dir = self.get_temp_dir()
        projects_dir = os.path.join(temp_dir, "projects_dir")
        os.mkdir(projects_dir)
        project_dir = os.path.join(projects_dir, "experiment_test")
        uri_dir = os.path.join(temp_dir, "env_uri")
        if name_from_meta:
            os.mkdir(project_dir)
            meta_fname = os.path.join(project_dir, "nucleus7_project.json")
            with open(meta_fname, "w") as f:
                json.dump({"PROJECT_NAME": "experiment_test_meta"}, f)

        if with_uri:
            os.environ["MLFLOW_TRACKING_URI"] = uri_dir
        mlflow_utils.create_new_or_continue_experiment(project_dir)
        if not mlflow.active_run():
            with mlflow.start_run():
                experiment_id = mlflow.active_run().info.experiment_id
        else:
            experiment_id = mlflow.active_run().info.experiment_id

        experiment_name = (
            mlflow.tracking.MlflowClient().get_experiment(experiment_id).name)
        experiment_name_must = (name_from_meta and "experiment_test_meta"
                                or "experiment_test")
        uri_must = with_uri and uri_dir or os.path.join(projects_dir, 'mlruns')
        self.assertEqual(uri_must,
                         mlflow.get_tracking_uri())
        self.assertEqual(experiment_name_must,
                         experiment_name)

    @parameterized.parameters({"with_active_run": True},
                              {"with_active_run": False})
    @patch("mlflow.active_run")
    @patch("mlflow.log_param")
    def test_log_config_parameter_to_mlflow(
            self, mlflow_log_param_fn, mlflow_active_run_fn, with_active_run):

        def _log_param(param_name, param_value):
            return

        def _active_run():
            if with_active_run:
                return True
            return None

        mlflow_log_param_fn.side_effect = _log_param
        mlflow_active_run_fn.side_effect = _active_run

        param_name = "param"
        param_value = "param_value"

        mlflow_utils.log_config_parameter_to_mlflow(param_name, param_value)
        if with_active_run:
            mlflow_log_param_fn.assert_called_once_with(
                param_name, param_value)
        else:
            mlflow_log_param_fn.assert_not_called()

    @parameterized.parameters({"with_active_run": True},
                              {"with_active_run": False})
    @patch("mlflow.active_run")
    @patch("mlflow.tracking.client.MlflowClient.log_metric")
    def test_log_metric_to_mlflow(
            self, mlflow_log_metric_fn, mlflow_active_run_fn, with_active_run):

        def _log_metric(param_name, param_value):
            return

        def _active_run():
            if with_active_run:
                info = mlflow_run_info.RunInfo(
                    run_uuid="run_uuidddd", experiment_id=1,
                    user_id="user_id", status="status", start_time=0,
                    end_time=None, lifecycle_stage=None, artifact_uri=None)
                new_run = mlflow_run.Run(info, None)
                return new_run
            return None

        mlflow_log_metric_fn.side_effect = _log_metric
        mlflow_active_run_fn.side_effect = _active_run

        param_name = "param"
        param_value = "param_value"

        mlflow_utils.log_metric_to_mlflow(param_name, param_value)
        if with_active_run:
            mlflow_log_metric_fn.assert_called_once_with(
                "run_uuidddd", param_name, param_value, ANY)
        else:
            mlflow_log_metric_fn.assert_not_called()

    @parameterized.parameters({"with_active_run": True},
                              {"with_active_run": False})
    @patch("mlflow.active_run")
    @patch("mlflow.tensorflow.log_model")
    @patch("mlflow.get_artifact_uri")
    @patch("nucleus7.utils.project_utils.log_exported_model_info")
    def test_log_saved_model(
            self, log_exported_model_info_fn, mlflow_get_artifact_uri_fn,
            mlflow_log_model_fn, mlflow_active_run_fn, with_active_run):
        def _log_model(tf_saved_model_dir, tf_meta_graph_tags,
                       tf_signature_def_key, artifact_path,
                       conda_env=None):
            return

        def _active_run():
            if with_active_run:
                return True
            return None

        def _get_artifact_uri():
            return "path/to/artifacts"

        def _log_exported_model_info(saved_model_artifact_path, global_step):
            return

        mlflow_log_model_fn.side_effect = _log_model
        mlflow_active_run_fn.side_effect = _active_run
        mlflow_get_artifact_uri_fn.side_effect = _get_artifact_uri
        log_exported_model_info_fn.side_effect = _log_exported_model_info

        saved_model_path = "path/to/model/tag_of_model"
        artifact_path_must = "models/tag_of_model"
        global_step = 456

        mlflow_utils.log_saved_model(saved_model_path, global_step)
        if with_active_run:
            mlflow_log_model_fn.assert_called_once_with(
                tf_saved_model_dir=saved_model_path,
                tf_meta_graph_tags=['serve'],
                tf_signature_def_key='serving_default',
                artifact_path=artifact_path_must
            )
            log_exported_model_info_fn.assert_called_once_with(
                "path/to/artifacts/models/tag_of_model",
                global_step)
        else:
            mlflow_log_model_fn.assert_not_called()

    def test_log_nucleotide_exec_time_to_mlflow(self):
        pass
