# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

import json
import os

from absl.testing import parameterized
import tensorflow as tf

from nucleus7.core import project_dirs
from nucleus7.core import project_global_config
from nucleus7.core.project_dirs import ProjectDirs
from nucleus7.core.project_dirs import create_data_extraction_project_dirs
from nucleus7.core.project_dirs import create_inference_project_dirs
from nucleus7.core.project_dirs import create_kpi_project_dirs
from nucleus7.core.project_dirs import create_trainer_project_dirs
from nucleus7.utils.io_utils import load_json


class TestProjectDirs(tf.test.TestCase, parameterized.TestCase):

    def test_create_trainer_project_dirs_local(self):
        if "TF_CONFIG" in os.environ:
            del os.environ["TF_CONFIG"]
        project_dir = self.get_temp_dir()
        project_file = os.path.join(project_dir, 'nucleus7_project.json')
        trainer_dirs = create_trainer_project_dirs(project_dir)
        self.assertIsInstance(trainer_dirs, tuple)
        self.assertTupleEqual(trainer_dirs._fields, ProjectDirs.TRAINER._fields)
        self.assertTrue(os.path.isfile(project_file))
        for k in trainer_dirs._fields:
            dir_must = os.path.join(project_dir,
                                    getattr(ProjectDirs.TRAINER, k))
            self.assertTrue(os.path.isdir(dir_must))

        self.assertTrue(os.path.isfile(project_file))
        project_meta_info = load_json(project_file)
        self.assertListEqual(['chief:0'], project_meta_info['runs'])

        with self.assertRaises(FileExistsError):
            _ = create_trainer_project_dirs(project_dir)

    def test_create_trainer_project_dirs_distributed(self):
        tasks = [{"type": "chief", "index": 0},
                 {"type": "worker", "index": 0},
                 {"type": "worker", "index": 1},
                 {"type": "ps", "index": 0},
                 {"type": "ps", "index": 1},
                 {"type": "evaluator", "index": 0}]
        cluster = {"chief": ["localhost:1111"],
                   "worker": ["localhost:2222", "localhost:3333"],
                   "ps": ["localhost:4444", "localhost:5555"]}

        project_dir = self.get_temp_dir()
        project_file = os.path.join(project_dir, 'nucleus7_project.json')
        for task in tasks:
            tf_config = {'cluster': cluster, "task": task}
            os.environ["TF_CONFIG"] = json.dumps(tf_config)
            _ = create_trainer_project_dirs(project_dir)
            config_dir = os.path.join(project_dir, ProjectDirs.TRAINER.configs)
            self.assertTrue(os.path.isdir(config_dir))

        self.assertTrue(os.path.isfile(project_file))
        project_meta_info = load_json(project_file)
        runs_must = ['{}:{}'.format(task['type'], task['index'])
                     for task in tasks]
        self.assertListEqual(runs_must, project_meta_info['runs'])

        for task in tasks:
            tf_config = {'cluster': cluster, "task": task}
            os.environ["TF_CONFIG"] = json.dumps(tf_config)
            if task['type'] == 'chief':
                with self.assertRaises(FileExistsError):
                    _ = create_trainer_project_dirs(project_dir)
            else:
                _ = create_trainer_project_dirs(project_dir)

        del os.environ['TF_CONFIG']
        with self.assertRaises(FileExistsError):
            _ = create_trainer_project_dirs(project_dir)

    def test_create_trainer_project_dirs_continue_training(self):
        if "TF_CONFIG" in os.environ:
            del os.environ["TF_CONFIG"]
        project_dir = self.get_temp_dir()
        project_file = os.path.join(project_dir, 'nucleus7_project.json')
        create_trainer_project_dirs(project_dir, continue_training=True)
        trainer_dirs = None
        for i in range(2):
            trainer_dirs = create_trainer_project_dirs(project_dir,
                                                       continue_training=True)

        for k in trainer_dirs._fields:
            dir_must = os.path.join(project_dir,
                                    getattr(ProjectDirs.TRAINER, k))
            self.assertTrue(os.path.isdir(dir_must))

        self.assertTrue(os.path.isfile(project_file))
        project_meta_info = load_json(project_file)
        runs_must = ['chief:0', 'chief:0/1', 'chief:0/2']
        self.assertListEqual(runs_must, project_meta_info['runs'])

        with self.assertRaises(FileExistsError):
            _ = create_trainer_project_dirs(project_dir)

    @parameterized.parameters({"continue_last": False, "run_name": None},
                              {"continue_last": True, "run_name": None},
                              {"continue_last": False, "run_name": "dummy"},
                              {"continue_last": True, "run_name": "dummy"})
    def test_create_inference_project_dirs_empty(self, continue_last,
                                                 run_name=None):
        project_dir = self.get_temp_dir()
        create_inference_project_dirs(project_dir,
                                      run_name=run_name,
                                      continue_last=continue_last)
        inference_dir = os.path.join(project_dir, "inference")
        run1_subdir_must = "run-1" if run_name is None else run_name
        run2_subdir_must = "run-2"
        last_run_link_path = os.path.join(inference_dir, "last_run")
        dirs_must = {
            inference_dir,
            os.path.join(inference_dir, "configs"),
            os.path.join(inference_dir, run1_subdir_must),
            os.path.join(inference_dir, run1_subdir_must, "results"),
            os.path.join(inference_dir, run1_subdir_must, "artifacts"),
            os.path.join(project_dir, "saved_models"),
            os.path.join(project_dir, "checkpoints"),
        }
        created_dirs = (
                {walk[0] for walk in os.walk(project_dir)} - {project_dir})
        self.assertSetEqual(dirs_must,
                            created_dirs)
        self.assertEqual(run1_subdir_must,
                         os.readlink(last_run_link_path))

        # should reuse run directory
        os.rmdir(os.path.join(inference_dir, run1_subdir_must, "results"))
        os.rmdir(os.path.join(inference_dir, run1_subdir_must, "artifacts"))
        create_inference_project_dirs(project_dir, run_name=run_name,
                                      continue_last=continue_last)
        created_dirs = (
                {walk[0] for walk in os.walk(project_dir)} - {project_dir})
        self.assertSetEqual(dirs_must,
                            created_dirs)
        self.assertEqual(run1_subdir_must,
                         os.readlink(last_run_link_path))

        # create new run directory if continue_last = False or reuse last one
        if run_name is not None and not continue_last:
            with self.assertRaises(ValueError):
                create_inference_project_dirs(project_dir, run_name=run_name,
                                              continue_last=continue_last)
            return

        create_inference_project_dirs(project_dir, run_name=run_name,
                                      continue_last=continue_last)
        if not continue_last:
            dirs_must.update({
                os.path.join(inference_dir, run2_subdir_must),
                os.path.join(inference_dir, run2_subdir_must, "results"),
                os.path.join(inference_dir, run2_subdir_must, "artifacts"),
            })
        created_dirs = (
                {walk[0] for walk in os.walk(project_dir)} - {project_dir})
        self.assertSetEqual(dirs_must,
                            created_dirs)
        if not continue_last:
            self.assertEqual("run-2",
                             os.readlink(last_run_link_path))

    @parameterized.parameters({"continue_last": False, "run_name": None},
                              {"continue_last": True, "run_name": None},
                              {"continue_last": False, "run_name": "dummy"},
                              {"continue_last": True, "run_name": "dummy"})
    def test_create_kpi_project_dirs_empty(self, continue_last, run_name=None):
        project_dir = self.get_temp_dir()
        create_kpi_project_dirs(project_dir,
                                run_name=run_name,
                                continue_last=continue_last)
        kpi_evaluation_dir = os.path.join(project_dir, "kpi_evaluation")
        run1_subdir_must = "run-1" if run_name is None else run_name
        run2_subdir_must = "run-2"
        last_run_link_path = os.path.join(kpi_evaluation_dir, "last_run")
        dirs_must = {
            kpi_evaluation_dir,
            os.path.join(kpi_evaluation_dir, "configs"),
            os.path.join(kpi_evaluation_dir, run1_subdir_must),
            os.path.join(kpi_evaluation_dir, run1_subdir_must, "results"),
            os.path.join(kpi_evaluation_dir, run1_subdir_must, "artifacts"),
        }
        created_dirs = (
                {walk[0] for walk in os.walk(project_dir)} - {project_dir})
        self.assertSetEqual(dirs_must,
                            created_dirs)
        self.assertEqual(run1_subdir_must,
                         os.readlink(last_run_link_path))

        # should reuse run directory
        os.rmdir(os.path.join(kpi_evaluation_dir, run1_subdir_must, "results"))
        os.rmdir(os.path.join(
            kpi_evaluation_dir, run1_subdir_must, "artifacts"))
        create_kpi_project_dirs(project_dir, run_name=run_name,
                                continue_last=continue_last)
        created_dirs = (
                {walk[0] for walk in os.walk(project_dir)} - {project_dir})
        self.assertSetEqual(dirs_must,
                            created_dirs)
        self.assertEqual(run1_subdir_must,
                         os.readlink(last_run_link_path))

        # create new run directory if continue_last is False otherwise use last
        # one
        if run_name is not None and not continue_last:
            with self.assertRaises(ValueError):
                create_kpi_project_dirs(project_dir, run_name=run_name,
                                        continue_last=continue_last)
            return

        create_kpi_project_dirs(project_dir, run_name=run_name,
                                continue_last=continue_last)
        if not continue_last:
            dirs_must.update({
                os.path.join(kpi_evaluation_dir, run2_subdir_must),
                os.path.join(kpi_evaluation_dir, run2_subdir_must, "results"),
                os.path.join(kpi_evaluation_dir, run2_subdir_must, "artifacts"),
            })
        created_dirs = (
                {walk[0] for walk in os.walk(project_dir)} - {project_dir})
        self.assertSetEqual(dirs_must,
                            created_dirs)
        if not continue_last:
            self.assertEqual("run-2",
                             os.readlink(last_run_link_path))

    @parameterized.parameters({"continue_last": False},
                              {"continue_last": True})
    def test_create_data_extraction_project_dirs_empty(self, continue_last):
        project_dir = self.get_temp_dir()
        create_data_extraction_project_dirs(project_dir, "subrun",
                                            continue_last=continue_last)
        data_extraction_dir = os.path.join(project_dir, "data_extraction")
        run1_subdir_must = "subrun"
        dirs_must = {
            data_extraction_dir,
            os.path.join(data_extraction_dir, "configs"),
            os.path.join(data_extraction_dir, run1_subdir_must),
            os.path.join(data_extraction_dir, run1_subdir_must, "extracted"),
            os.path.join(data_extraction_dir, run1_subdir_must, "artifacts"),
        }
        created_dirs = (
                {walk[0] for walk in os.walk(project_dir)} - {project_dir})
        self.assertSetEqual(dirs_must,
                            created_dirs)

        # should reuse run directory
        os.rmdir(os.path.join(
            data_extraction_dir, run1_subdir_must, "extracted"))
        os.rmdir(os.path.join(
            data_extraction_dir, run1_subdir_must, "artifacts"))
        create_data_extraction_project_dirs(project_dir, "subrun",
                                            continue_last=continue_last)
        created_dirs = (
                {walk[0] for walk in os.walk(project_dir)} - {project_dir})
        self.assertSetEqual(dirs_must,
                            created_dirs)

        if continue_last:
            create_data_extraction_project_dirs(project_dir, "subrun",
                                                continue_last=continue_last)
        else:
            with self.assertRaises(ValueError):
                create_data_extraction_project_dirs(project_dir, "subrun",
                                                    continue_last=continue_last)

        # create new run directory
        create_data_extraction_project_dirs(project_dir, "subrun2",
                                            continue_last=continue_last)
        dirs_must.update({
            os.path.join(data_extraction_dir, "subrun2"),
            os.path.join(data_extraction_dir, "subrun2", "extracted"),
            os.path.join(data_extraction_dir, "subrun2", "artifacts"),
        })
        created_dirs = (
                {walk[0] for walk in os.walk(project_dir)} - {project_dir})
        self.assertSetEqual(dirs_must,
                            created_dirs)

    def test_read_train_configs_from_directories(self):
        project_dir = self.get_temp_dir()
        project_global_config.clear_project_global_config()

        subdir = project_dirs._TRAINING_DIR
        configs_dir = os.path.join(project_dir, subdir, "configs")
        os.makedirs(configs_dir)
        single_names = [
            each_name
            for each_name in project_dirs._TRAINING_FILE_NAMES_WITH_CONFIGS
            if each_name != "model"]
        _write_single_configs(configs_dir, single_names)
        config_must = {k + "_config": {"data": k} for k in single_names}

        config_dir_names = project_dirs._TRAINING_DIRECTORIES_WITH_CONFIGS
        for each_dir in config_dir_names:
            config_dir = os.path.join(configs_dir, each_dir)
            os.mkdir(config_dir)
            single_names = [each_dir + "_%s" % i for i in range(2)]
            _write_single_configs(config_dir, single_names)
            config_must.update(
                {each_dir + "_config": [{"data": k} for k in single_names]})

        config_main = {'model_config': {'data': 'model'},
                       "trainer_config": {"data": "trainer_from_main_config"}}
        config_must.update({'model_config': {'data': 'model'}})
        _write_config_main(configs_dir, config_main)
        _write_global_config(configs_dir)

        wrong_config_name = "wrong_config"
        _write_config(configs_dir, wrong_config_name, {})
        not_used_config_name = "_not_used_config"
        _write_config(configs_dir, not_used_config_name, {})

        with self.assertRaises(FileExistsError):
            _ = project_dirs.read_train_configs_from_directories(
                project_dir, verify_unused_configs=True)
        os.remove(os.path.join(configs_dir, wrong_config_name + ".json"))
        os.mkdir(os.path.join(configs_dir, "wrong_dir"))
        with self.assertRaises(FileExistsError):
            _ = project_dirs.read_train_configs_from_directories(
                project_dir, verify_unused_configs=True)
        os.rmdir(os.path.join(configs_dir, "wrong_dir"))

        config = project_dirs.read_train_configs_from_directories(
            project_dir, verify_unused_configs=True)

        self.assertDictEqual(config_must,
                             config)
        self.assertDictEqual({"p1": "global_p1", "p2": "global_p2"},
                             project_global_config._ProjectGlobalConfig.config)

    @parameterized.parameters(
        {"with_main_config": True, "with_run_config": True,
         "with_additional_config": True},
        {"with_main_config": True, "with_run_config": True,
         "with_additional_config": True, "continue_last": True},
        {"with_main_config": True, "with_run_config": True,
         "run_name": "dummy",
         "with_additional_config": True},
        {"with_main_config": True, "with_run_config": True,
         "run_name": "dummy",
         "with_additional_config": True, "continue_last": True},
        {"with_main_config": True, "with_run_config": True,
         "with_additional_config": False},
        {"with_main_config": False, "with_run_config": False,
         "with_additional_config": False},
        {"with_main_config": False, "with_run_config": True,
         "with_additional_config": False},
        {"with_main_config": False, "with_run_config": True,
         "with_additional_config": False, "continue_last": True},
        {"with_main_config": False, "with_run_config": False,
         "with_additional_config": False, "continue_last": True}
    )
    def test_read_inference_configs_from_directories(
            self, with_main_config, with_run_config, with_additional_config,
            run_name=None,
            continue_last=False):
        project_dir = self.get_temp_dir()
        project_global_config.clear_project_global_config()
        subdir = project_dirs._INFERENCE_DIR
        config_must = {}
        global_config_must = {}
        configs_dir = os.path.join(project_dir, subdir, "configs")
        if with_main_config:
            os.makedirs(configs_dir)
            single_names = [
                each_name for each_name in
                project_dirs._INFERENCE_FILE_NAMES_WITH_CONFIGS]
            _write_single_configs(configs_dir, single_names,
                                  name_prefix="global")
            _write_config(configs_dir, "_not_used_config", {"data": "not_used"})
            config_must.update({k + "_config": {"data": "global_" + k}
                                for k in single_names})
            config_dir_names = project_dirs._INFERENCE_DIRECTORIES_WITH_CONFIGS
            for each_dir in config_dir_names:
                config_dir = os.path.join(configs_dir, each_dir)
                os.mkdir(config_dir)
                single_names = [each_dir + "_%s" % i for i in range(2)]
                _write_single_configs(config_dir, single_names,
                                      name_prefix="global")
                config_must.update(
                    {each_dir + "_config": [{"data": "global_" + k}
                                            for k in single_names]})
                _write_global_config(configs_dir)
                global_config_must.update(
                    {"p1": "global_p1", "p2": "global_p2"})

        if with_run_config:
            run_dir = run_name or "run-1"
            run_configs_dir = os.path.join(
                project_dir, subdir, run_dir, "configs")
            os.makedirs(run_configs_dir)
            single_names = ["datafeeder"]
            _write_single_configs(run_configs_dir, single_names,
                                  name_prefix="run")
            config_must.update({k + "_config": {"data": "run_" + k}
                                for k in single_names})
            callbacks_run_config_dir = os.path.join(
                run_configs_dir, "callbacks")
            os.mkdir(callbacks_run_config_dir)
            _write_config(callbacks_run_config_dir, "base", ["__BASE__"])
            _write_config(callbacks_run_config_dir, "callback_run",
                          {"data": "callback_run_data"})
            config_must.setdefault("callbacks_config", [])
            if not with_main_config:
                config_must["callbacks_config"].append("__BASE__")
            config_must["callbacks_config"].append(
                {"data": "callback_run_data"})

            _write_global_config(run_configs_dir,
                                 {"p1": "run_p1", "p3": "run_p3"})
            global_config_must.update({"p1": "run_p1", "p3": "run_p3"})

            if continue_last:
                results_run_config_dir = os.path.join(
                    project_dir, subdir, "results")
                os.mkdir(results_run_config_dir)

        if with_additional_config:
            additional_configs_dir = os.path.join(
                project_dir, "additional_configs")
            os.mkdir(additional_configs_dir)
            kpi_additional_config_dir = os.path.join(
                additional_configs_dir, "kpi")
            os.mkdir(kpi_additional_config_dir)
            _write_config(kpi_additional_config_dir, "kpi_additional",
                          {"data": "kpi_additional"})
            config_must["kpi_config"] = [{"data": "kpi_additional"}]

            _write_global_config(additional_configs_dir,
                                 {"p4": "add_p4", "p2": "add_p3"})
            global_config_must.update({"p4": "add_p4", "p2": "add_p3"})
        else:
            additional_configs_dir = None

        if (not with_main_config
                and not with_run_config
                and not with_additional_config):
            with self.assertRaises(FileNotFoundError):
                project_dirs.read_inference_configs_from_directories(
                    project_dir, run_name=run_name, verify_unused_configs=True,
                    additional_dir_with_configs=additional_configs_dir,
                    continue_last=continue_last)
            return

        if with_main_config:
            os.mkdir(os.path.join(configs_dir, "wrong_config_dir"))
            with self.assertRaises(FileExistsError):
                project_dirs.read_inference_configs_from_directories(
                    project_dir, run_name=run_name, verify_unused_configs=True,
                    additional_dir_with_configs=additional_configs_dir,
                    continue_last=continue_last)
            os.rmdir(os.path.join(configs_dir, "wrong_config_dir"))

        config = project_dirs.read_inference_configs_from_directories(
            project_dir, run_name=run_name, verify_unused_configs=True,
            additional_dir_with_configs=additional_configs_dir,
            continue_last=continue_last)
        self.assertDictEqual(config_must,
                             config)
        self.assertDictEqual(global_config_must,
                             project_global_config._ProjectGlobalConfig.config)

    @parameterized.parameters(
        {"with_main_config": True, "with_run_config": True,
         "with_additional_config": True},
        {"with_main_config": True, "with_run_config": True,
         "with_additional_config": True, "continue_last": True},
        {"with_main_config": True, "with_run_config": True,
         "run_name": "dummy",
         "with_additional_config": True},
        {"with_main_config": True, "with_run_config": True,
         "run_name": "dummy",
         "with_additional_config": True, "continue_last": True},
        {"with_main_config": True, "with_run_config": True,
         "with_additional_config": False},
        {"with_main_config": False, "with_run_config": False,
         "with_additional_config": False},
        {"with_main_config": False, "with_run_config": True,
         "with_additional_config": False},
        {"with_main_config": False, "with_run_config": True,
         "with_additional_config": False, "continue_last": True},
        {"with_main_config": False, "with_run_config": False,
         "with_additional_config": False, "continue_last": True}
    )
    def test_read_kpi_configs_from_directories(
            self, with_main_config, with_run_config, with_additional_config,
            run_name=None,
            continue_last=False):
        project_dir = self.get_temp_dir()
        project_global_config.clear_project_global_config()
        subdir = project_dirs._KPI_DIR
        config_must = {}
        global_config_must = {}
        configs_dir = os.path.join(project_dir, subdir, "configs")
        if with_main_config:
            os.makedirs(configs_dir)
            single_names = [
                each_name for each_name in
                project_dirs._KPI_FILE_NAMES_WITH_CONFIGS]
            _write_single_configs(configs_dir, single_names,
                                  name_prefix="global")
            _write_config(configs_dir, "_not_used_config", {"data": "not_used"})

            config_must.update({k + "_config": {"data": "global_" + k}
                                for k in single_names})
            config_dir_names = project_dirs._KPI_DIRECTORIES_WITH_CONFIGS
            for each_dir in config_dir_names:
                config_dir = os.path.join(configs_dir, each_dir)
                os.mkdir(config_dir)
                single_names = [each_dir + "_%s" % i for i in range(2)]
                _write_single_configs(config_dir, single_names,
                                      name_prefix="global")
                config_must.update(
                    {each_dir + "_config": [{"data": "global_" + k}
                                            for k in single_names]})
                _write_global_config(configs_dir)
                global_config_must.update(
                    {"p1": "global_p1", "p2": "global_p2"})

        if with_run_config:
            run_dir = run_name or "run-1"
            run_configs_dir = os.path.join(
                project_dir, subdir, run_dir, "configs")
            os.makedirs(run_configs_dir)
            single_names = ["datafeeder"]
            _write_single_configs(run_configs_dir, single_names,
                                  name_prefix="run")
            config_must.update({k + "_config": {"data": "run_" + k}
                                for k in single_names})
            callbacks_run_config_dir = os.path.join(
                run_configs_dir, "callbacks")
            os.mkdir(callbacks_run_config_dir)
            _write_config(callbacks_run_config_dir, "base", ["__BASE__"])
            _write_config(callbacks_run_config_dir, "callback_run",
                          {"data": "callback_run_data"})
            config_must.setdefault("callbacks_config", [])
            if not with_main_config:
                config_must["callbacks_config"].append("__BASE__")
            config_must["callbacks_config"].append(
                {"data": "callback_run_data"})

            _write_global_config(run_configs_dir,
                                 {"p1": "run_p1", "p3": "run_p3"})
            global_config_must.update({"p1": "run_p1", "p3": "run_p3"})
            if continue_last:
                results_run_config_dir = os.path.join(
                    project_dir, subdir, "results")
                os.mkdir(results_run_config_dir)

        if with_additional_config:
            additional_configs_dir = os.path.join(
                project_dir, "additional_configs")
            os.mkdir(additional_configs_dir)
            kpi_additional_config_dir = os.path.join(
                additional_configs_dir, "kpi")
            os.mkdir(kpi_additional_config_dir)
            _write_config(kpi_additional_config_dir, "kpi_additional",
                          {"data": "kpi_additional"})
            config_must["kpi_config"] = [{"data": "kpi_additional"}]

            _write_global_config(additional_configs_dir,
                                 {"p4": "add_p4", "p2": "add_p3"})
            global_config_must.update({"p4": "add_p4", "p2": "add_p3"})
        else:
            additional_configs_dir = None

        if (not with_main_config
                and not with_run_config
                and not with_additional_config):
            with self.assertRaises(FileNotFoundError):
                project_dirs.read_kpi_configs_from_directories(
                    project_dir, run_name=run_name, verify_unused_configs=True,
                    additional_dir_with_configs=additional_configs_dir,
                    continue_last=continue_last)
            return

        if with_main_config:
            os.mkdir(os.path.join(configs_dir, "wrong_config_dir"))
            with self.assertRaises(FileExistsError):
                project_dirs.read_kpi_configs_from_directories(
                    project_dir, run_name=run_name, verify_unused_configs=True,
                    additional_dir_with_configs=additional_configs_dir)
            os.rmdir(os.path.join(configs_dir, "wrong_config_dir"))

        config = project_dirs.read_kpi_configs_from_directories(
            project_dir, run_name=run_name, verify_unused_configs=True,
            additional_dir_with_configs=additional_configs_dir,
            continue_last=continue_last)

        self.assertDictEqual(config_must,
                             config)
        self.assertDictEqual(global_config_must,
                             project_global_config._ProjectGlobalConfig.config)

    @parameterized.parameters(
        {"with_main_config": True, "with_run_config": True,
         "with_additional_config": True},
        {"with_main_config": True, "with_run_config": True,
         "with_additional_config": True, "continue_last": True},
        {"with_main_config": True, "with_run_config": True,
         "with_additional_config": False},
        {"with_main_config": False, "with_run_config": False,
         "with_additional_config": False},
        {"with_main_config": False, "with_run_config": True,
         "with_additional_config": False},
        {"with_main_config": False, "with_run_config": True,
         "with_additional_config": False, "continue_last": True},
        {"with_main_config": False, "with_run_config": False,
         "with_additional_config": False, "continue_last": True}
    )
    def test_read_data_extraction_configs_from_directories(
            self, with_main_config, with_run_config, with_additional_config,
            continue_last=False):
        project_dir = self.get_temp_dir()
        run_name = "run_name"
        project_global_config.clear_project_global_config()
        subdir = project_dirs._DATA_EXTRACT_DIR
        config_must = {}
        global_config_must = {}
        configs_dir = os.path.join(project_dir, subdir, "configs")
        if with_main_config:
            os.makedirs(configs_dir)
            single_names = [
                each_name for each_name in
                project_dirs._DATA_EXTRACT_FILE_NAMES_WITH_CONFIGS]
            _write_single_configs(configs_dir, single_names,
                                  name_prefix="global")
            _write_config(configs_dir, "_not_used_config", {"data": "not_used"})

            config_must.update({k + "_config": {"data": "global_" + k}
                                for k in single_names})
            config_dir_names = (
                project_dirs._DATA_EXTRACT_DIRECTORIES_WITH_CONFIGS)
            for each_dir in config_dir_names:
                config_dir = os.path.join(configs_dir, each_dir)
                os.mkdir(config_dir)
                single_names = [each_dir + "_%s" % i for i in range(2)]
                _write_single_configs(config_dir, single_names,
                                      name_prefix="global")
                config_must.update(
                    {each_dir + "_config": [{"data": "global_" + k}
                                            for k in single_names]})
                _write_global_config(configs_dir)
                global_config_must.update(
                    {"p1": "global_p1", "p2": "global_p2"})

        if with_run_config:
            run_dir = run_name
            run_configs_dir = os.path.join(
                project_dir, subdir, run_dir, "configs")
            os.makedirs(run_configs_dir)
            single_names = ["datafeeder"]
            _write_single_configs(run_configs_dir, single_names,
                                  name_prefix="run")
            config_must.update({k + "_config": {"data": "run_" + k}
                                for k in single_names})
            callbacks_run_config_dir = os.path.join(
                run_configs_dir, "callbacks")
            os.mkdir(callbacks_run_config_dir)
            _write_config(callbacks_run_config_dir, "base", ["__BASE__"])
            _write_config(callbacks_run_config_dir, "callback_run",
                          {"data": "callback_run_data"})
            config_must.setdefault("callbacks_config", [])
            if not with_main_config:
                config_must["callbacks_config"].append("__BASE__")
            config_must["callbacks_config"].append(
                {"data": "callback_run_data"})

            _write_global_config(run_configs_dir,
                                 {"p1": "run_p1", "p3": "run_p3"})
            global_config_must.update({"p1": "run_p1", "p3": "run_p3"})
            if continue_last:
                results_run_config_dir = os.path.join(
                    project_dir, subdir, "results")
                os.mkdir(results_run_config_dir)

        if with_additional_config:
            additional_configs_dir = os.path.join(
                project_dir, "additional_configs")
            os.mkdir(additional_configs_dir)
            callbacks_additional_config_dir = os.path.join(
                additional_configs_dir, "callbacks")
            os.mkdir(callbacks_additional_config_dir)
            _write_config(callbacks_additional_config_dir,
                          "callback_additional",
                          {"data": "callback_additional"})
            config_must["callbacks_config"] = [{"data": "callback_additional"}]

            _write_global_config(additional_configs_dir,
                                 {"p4": "add_p4", "p2": "add_p3"})
            global_config_must.update({"p4": "add_p4", "p2": "add_p3"})
        else:
            additional_configs_dir = None

        if (not with_main_config
                and not with_run_config
                and not with_additional_config):
            with self.assertRaises(FileNotFoundError):
                project_dirs.read_data_extraction_configs_from_directories(
                    project_dir, run_name, verify_unused_configs=True,
                    additional_dir_with_configs=additional_configs_dir,
                    continue_last=continue_last)
            return

        if with_main_config:
            os.mkdir(os.path.join(configs_dir, "wrong_config_dir"))
            with self.assertRaises(FileExistsError):
                project_dirs.read_data_extraction_configs_from_directories(
                    project_dir, run_name, verify_unused_configs=True,
                    additional_dir_with_configs=additional_configs_dir,
                    continue_last=continue_last)
            os.rmdir(os.path.join(configs_dir, "wrong_config_dir"))

        config = project_dirs.read_data_extraction_configs_from_directories(
            project_dir, run_name, verify_unused_configs=True,
            additional_dir_with_configs=additional_configs_dir,
            continue_last=continue_last)

        self.assertDictEqual(config_must,
                             config)
        self.assertDictEqual(global_config_must,
                             project_global_config._ProjectGlobalConfig.config)


def _write_single_configs(directory, names, name_prefix=None):
    for each_name in names:
        content = ({"data": each_name} if not name_prefix
                   else {"data": "_".join([name_prefix, each_name])})
        _write_config(directory, each_name, content)


def _write_config(directory, name, content):
    full_fname = os.path.join(directory, name + ".json")
    with open(full_fname, "w") as f:
        json.dump(content, f)


def _write_global_config(directory, content=None):
    content = content or {"p1": "global_p1", "p2": "global_p2"}
    _write_config(directory, "project_global_config", content)


def _write_config_main(directory, content):
    _write_config(directory, "config_main", content)
