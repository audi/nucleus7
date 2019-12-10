# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

import os
import tempfile

from absl.testing import parameterized

import nucleus7.coordinator.configs as cconfigs
from nucleus7.core.project_dirs import ProjectDirs
from nucleus7.test_utils.test_utils import TestCaseWithTempDir
from nucleus7.utils.io_utils import maybe_mkdir


class TestCoordinatorConfigs(parameterized.TestCase, TestCaseWithTempDir):

    def test_create_and_validate_trainer_run_config(self):
        batch_size = {'train': 10, 'eval': 2}
        devices = ['dev:1', 'dev:2']
        variable_strategy = 'CPU'
        iterations_per_epoch = {'train': 200}
        samples_per_epoch = {'eval': 100}
        predictions_have_variable_shape = True
        continue_training = False
        profile_hook_config = {'save_secs': 10}
        config = cconfigs.create_and_validate_trainer_run_config(
            batch_size=batch_size,
            devices=devices,
            samples_per_epoch=samples_per_epoch,
            iterations_per_epoch=iterations_per_epoch,
            variable_strategy=variable_strategy,
            predictions_have_variable_shape=predictions_have_variable_shape,
            continue_training=continue_training,
            profile_hook_config=profile_hook_config
        )
        self.assertIsInstance(config, cconfigs.TrainerRunConfig)

        self.assertDictEqual(config.batch_size, {'train': 20, 'eval': 4})
        self.assertListEqual(config.devices, devices)
        self.assertEqual(config.variable_strategy, variable_strategy)
        self.assertDictEqual(config.iterations_per_epoch,
                             {'train': 200, 'eval': 25})
        self.assertEqual(config.predictions_have_variable_shape,
                         predictions_have_variable_shape)
        self.assertEqual(config.continue_training, continue_training)
        self.assertEqual(config.profile_hook_config, profile_hook_config)

        batch_size = 5
        iterations_per_epoch = {'train': 100}
        samples_per_epoch = {'eval': 50}
        config = cconfigs.create_and_validate_trainer_run_config(
            batch_size=batch_size,
            devices=devices,
            samples_per_epoch=samples_per_epoch,
            iterations_per_epoch=iterations_per_epoch
        )
        self.assertDictEqual(config.batch_size, {'train': 10, 'eval': 10})
        self.assertDictEqual(config.iterations_per_epoch,
                             {'train': 100, 'eval': 5})
        self.assertIsNone(config.profile_hook_config)

        iterations_per_epoch = {'train': 100, 'eval': 50}
        samples_per_epoch = {'eval': 50}
        with self.assertRaises(AssertionError):
            _ = cconfigs.create_and_validate_trainer_run_config(
                batch_size=batch_size,
                devices=devices,
                samples_per_epoch=samples_per_epoch,
                iterations_per_epoch=iterations_per_epoch
            )
        with self.assertRaises(AssertionError):
            _ = cconfigs.create_and_validate_trainer_run_config(
                batch_size=batch_size,
                devices=devices,
                samples_per_epoch=None,
                iterations_per_epoch={'train': 100}
            )
        with self.assertRaises(AssertionError):
            _ = cconfigs.create_and_validate_trainer_run_config(
                batch_size=batch_size,
                devices=devices,
                samples_per_epoch={'eval': 100},
                iterations_per_epoch=None
            )

        iterations_per_epoch = {'train': 100, 'eval': 50}
        config = cconfigs.create_and_validate_trainer_run_config(
            batch_size=batch_size,
            devices=devices,
            samples_per_epoch=None,
            iterations_per_epoch=iterations_per_epoch
        )
        self.assertDictEqual(config.iterations_per_epoch,
                             {'train': 100, 'eval': 50})

        batch_size = {'train': 10, 'eval': 2}
        samples_per_epoch = {'train': 200, 'eval': 20}
        config = cconfigs.create_and_validate_trainer_run_config(
            batch_size=batch_size,
            devices=devices,
            samples_per_epoch=samples_per_epoch,
            iterations_per_epoch=None
        )
        self.assertDictEqual(config.iterations_per_epoch,
                             {'train': 10, 'eval': 5})

    def test_create_and_validate_trainer_save_config(self):
        inference_inputs_have_variable_shape = False
        exports_to_keep = 10
        save_summary_steps = {'train': 100, 'eval': 10}
        max_outputs_tb = 3

        config = cconfigs.create_and_validate_trainer_save_config(
            inference_inputs_have_variable_shape=inference_inputs_have_variable_shape,
            exports_to_keep=exports_to_keep,
            save_summary_steps=save_summary_steps,
            max_outputs_tb=max_outputs_tb
        )
        self.assertIsInstance(config, cconfigs.TrainerSaveConfig)
        self.assertDictEqual(config.save_summary_steps, save_summary_steps)
        self.assertEqual(config.exports_to_keep, exports_to_keep)
        self.assertEqual(config.inference_inputs_have_variable_shape,
                         inference_inputs_have_variable_shape)
        self.assertEqual(config.max_outputs_tb, max_outputs_tb)

        with self.assertRaises(AssertionError):
            _ = cconfigs.create_and_validate_trainer_save_config(
                save_summary_steps={'train': 100}
            )
        with self.assertRaises(AssertionError):
            _ = cconfigs.create_and_validate_trainer_save_config(
                save_summary_steps={'eval': 100}
            )

        config = cconfigs.create_and_validate_trainer_save_config(
            save_summary_steps=10
        )
        self.assertDictEqual(config.save_summary_steps,
                             {'train': 10, 'eval': 10})

    @parameterized.parameters({'use_recent': False},
                              {'use_recent': True})
    def test_create_and_validate_inference_load_config_from_saved_models(
            self, use_recent):
        project_dir = self.get_temp_dir()
        saved_models_dir = os.path.join(
            project_dir, ProjectDirs.TRAINER.saved_models)
        maybe_mkdir(saved_models_dir)
        saved_models_tag_dirs = [os.path.join(saved_models_dir, sd)
                                 for sd in ['first', 'second', 'third']]
        for d in saved_models_tag_dirs:
            maybe_mkdir(d)

        if use_recent:
            saved_model_path = None
            saved_model_path_must = saved_models_tag_dirs[-1]
        else:
            saved_model_path = "first"
            saved_model_path_must = saved_models_tag_dirs[0]

        config = cconfigs.create_and_validate_inference_load_config(
            project_dir=project_dir, saved_model_path=saved_model_path)
        self.assertIsInstance(config, cconfigs.InferenceLoadConfig)
        self.assertEqual(config.saved_model, saved_model_path_must)
        self.assertIsNone(config.checkpoint)
        self.assertIsNone(config.meta_graph)

        with self.assertRaises(ValueError):
            _ = cconfigs.create_and_validate_inference_load_config(
                project_dir=project_dir, saved_model_path="path_without_file",
                checkpoint_path="path_without_file")
        with self.assertRaises(FileNotFoundError):
            _ = cconfigs.create_and_validate_inference_load_config(
                project_dir=project_dir, saved_model_path="path_without_file")

    def test_create_and_validate_inference_load_config_from_graph(self):
        project_dir = self.get_temp_dir()
        checkpoints_dir = os.path.join(project_dir, "checkpoints")
        os.mkdir(checkpoints_dir)

        _, checkpoint_path = tempfile.mkstemp(
            suffix='.index', dir=checkpoints_dir)
        checkpoint_path = checkpoint_path.split('.index')[0]
        relative_checkpoint_path = os.path.relpath(
            checkpoint_path, checkpoints_dir)

        meta_graph_path = os.path.join(
            project_dir, "checkpoints", "graph_inference.meta")
        with open(meta_graph_path, "a") as f:
            f.close()

        config = cconfigs.create_and_validate_inference_load_config(
            project_dir=project_dir,
            checkpoint_path=relative_checkpoint_path)
        self.assertIsInstance(config, cconfigs.InferenceLoadConfig)

        self.assertIsNone(config.saved_model)
        self.assertEqual(config.meta_graph, meta_graph_path)
        self.assertEqual(config.checkpoint, checkpoint_path)

        with self.assertRaises(ValueError):
            _ = cconfigs.create_and_validate_inference_load_config(
                project_dir=project_dir,
                checkpoint_path=checkpoint_path,
                saved_model_path='path_without_file')
        with self.assertRaises(FileNotFoundError):
            _ = cconfigs.create_and_validate_inference_load_config(
                project_dir=project_dir,
                checkpoint_path="path_without_file")
