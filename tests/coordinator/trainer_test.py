# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

import os
from unittest.mock import MagicMock

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from nucleus7.coordinator import configs as cconfigs
from nucleus7.coordinator.callback import CoordinatorCallback
from nucleus7.coordinator.callbacks_handler import CallbacksHandler
from nucleus7.coordinator.trainer import Trainer
from nucleus7.core.project_dirs import ProjectDirs
from nucleus7.data.dataset import Dataset
from nucleus7.model.model import Model
from nucleus7.optimization.configs import (
    create_and_validate_optimization_config)
from nucleus7.test_utils.model_dummies import DummyPluginCNN
from nucleus7.test_utils.model_dummies import DummyPluginFlatten
from nucleus7.test_utils.model_dummies import DummyPluginMLP
from nucleus7.test_utils.model_dummies import DummySoftmaxLoss
from nucleus7.test_utils.test_utils import reset_register_and_logger


class TestTrainer(parameterized.TestCase, tf.test.TestCase):

    def setUp(self):
        reset_register_and_logger()

    def test_build(self):
        tf.reset_default_graph()
        batch_size = {'train': 4, 'eval': 2}
        samples_per_epoch = {'train': 100, 'eval': 100}
        iters_per_epoch = {k: v // batch_size[k]
                           for k, v in samples_per_epoch.items()}
        project_dir = self.get_temp_dir()

        model = self._get_model()
        callbacks_handler_train = self._get_callbacks_handler('train')
        callbacks_handler_eval = self._get_callbacks_handler('eval')

        datasets = {k: self._get_dataset() for k in ['train', 'eval']}
        optimization_config = create_and_validate_optimization_config(
            optimizer_name='rmsprop', learning_rate=1e-3, is_global=True
        )
        run_config = cconfigs.create_and_validate_trainer_run_config(
            batch_size=batch_size, samples_per_epoch=samples_per_epoch)
        save_config = cconfigs.create_and_validate_trainer_save_config()
        trainer = Trainer(model=model,
                          project_dir=project_dir,
                          datasets=datasets,
                          run_config=run_config,
                          save_config=save_config,
                          optimization_config=optimization_config,
                          callbacks_handler_train=callbacks_handler_train,
                          callbacks_handler_eval=callbacks_handler_eval)
        trainer.build()
        self.assertTrue(isinstance(trainer.estimator, tf.estimator.Estimator))
        self.assertTrue(isinstance(trainer.estimator_train_spec,
                                   tf.estimator.TrainSpec))
        self.assertTrue(isinstance(trainer.estimator_eval_spec,
                                   tf.estimator.EvalSpec))

        trainer.model.build_dna.assert_called_once_with(
            incoming_nucleotides=datasets['train'])

        for mode in ['train', 'eval']:
            callbacks_handler = trainer.callbacks_handler[mode]
            build_dna_handler_args, build_dna_handler_kwargs = (
                callbacks_handler.build_dna.call_args)
            self.assertEqual(0, len(build_dna_handler_args))
            self.assertSetEqual({'incoming_nucleotides'},
                                set(build_dna_handler_kwargs))
            incoming_nucleotides = set(
                build_dna_handler_kwargs['incoming_nucleotides'])
            incoming_nucleotides_must = list(
                trainer.model.all_nucleotides.values())
            incoming_nucleotides_must.append(datasets['train'])
            self.assertSetEqual(set(incoming_nucleotides_must),
                                incoming_nucleotides)
            self.assertEqual(mode, callbacks_handler.mode)
            self.assertEqual(os.path.join(trainer.project_dirs.callbacks, mode),
                             callbacks_handler.log_dir)
            self.assertEqual(iters_per_epoch[mode],
                             callbacks_handler.number_iterations_per_epoch)
        self.assertTrue(os.path.isdir(os.path.join(
            project_dir, ProjectDirs.TRAINER.callbacks)))
        self.assertTrue(os.path.isdir(os.path.join(
            project_dir, ProjectDirs.TRAINER.summaries)))
        self.assertTrue(os.path.isdir(os.path.join(
            project_dir, ProjectDirs.TRAINER.saved_models)))
        self.assertTrue(os.path.isdir(os.path.join(
            project_dir, ProjectDirs.TRAINER.checkpoints)))

    @staticmethod
    def _get_dataset():
        np.random.seed(546547)
        dataset_size = 200
        inputs_np = {
            'data': np.random.randn(dataset_size, 100).astype(np.float32),
            'labels': np.random.randint(
                10, size=(dataset_size,)).astype(np.int64),
            'temp': np.ones(dataset_size, np.float32)}

        def read_data_element():
            data = tf.data.Dataset.from_tensor_slices(inputs_np)
            return data

        dataset = Dataset().build()
        dataset.create_initial_data = MagicMock(side_effect=read_data_element)
        return dataset

    @staticmethod
    def _get_callbacks_handler(mode: str):
        callback1 = CoordinatorCallback(name='callback1',
                                        inbound_nodes=['dataset', 'mlp'])
        callback1.incoming_keys = ['image', 'predictions']
        callback1.generated_keys = ['result']

        callbacks = [callback1]
        if mode == 'train':
            callback2 = CoordinatorCallback(name='callback2',
                                            inbound_nodes=['dataset',
                                                           'callback1'])
            callback2.incoming_keys = ['image', 'result']
            callbacks.append(callback2)

        callbacks_handler = CallbacksHandler(
            callbacks=callbacks).build()
        callbacks_handler.build_dna = MagicMock(return_value=None)
        callbacks_handler._all_nucleotides = {}
        for gene_name in callbacks_handler.gene_name_and_nucleotide_super_cls:
            callbacks_handler.all_nucleotides.update(
                getattr(callbacks_handler, gene_name))
        return callbacks_handler

    def _get_model(self):
        plugins = self._get_model_plugins()
        losses = self._get_model_loss()
        model = Model(plugins=plugins,
                      losses=losses).build()
        model.build_dna = MagicMock(return_value=None)
        model._all_nucleotides = {}
        for gene_name in model.gene_name_and_nucleotide_super_cls:
            model.all_nucleotides.update(getattr(model, gene_name))
        return model

    @staticmethod
    def _get_model_plugins():
        plugin_cnn = DummyPluginCNN(
            name='cnn', inbound_nodes=['dataset'],
            incoming_keys_mapping={'dataset': {'image': 'inputs_cnn'}}).build()
        plugin_flatten = DummyPluginFlatten(
            name='flatten', inbound_nodes=['cnn'],
            incoming_keys_mapping={
                'cnn': {'predictions': 'inputs_flatten'}}).build()
        plugin_mlp = DummyPluginMLP(
            name='mlp', inbound_nodes=['flatten'],
            incoming_keys_mapping={
                'flatten': {'predictions': 'inputs_mlp'}},
            allow_mixed_precision=False).build()
        model_plugins = [plugin_cnn, plugin_flatten, plugin_mlp]
        return model_plugins

    @staticmethod
    def _get_model_loss():
        model_loss = DummySoftmaxLoss(
            inbound_nodes=['dataset', 'mlp'],
            incoming_keys_mapping={'mlp': {'predictions': 'logits'}}).build()
        return model_loss
