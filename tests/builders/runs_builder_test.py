# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Testing of run builders
"""

from unittest.mock import patch

from absl.testing import parameterized
import tensorflow as tf

import nucleus7 as nc7
from nucleus7.builders import runs_builder
from nucleus7.coordinator.configs import InferenceLoadConfig
from nucleus7.core.dna_helix import DNAHelix
from nucleus7.test_utils.model_dummies import FileListDummy
from nucleus7.test_utils.test_utils import register_new_class
from nucleus7.test_utils.test_utils import reset_register_and_logger


class _DatasetFileListDummy(nc7.data.DatasetFileList):
    file_list_keys = ['data']


class _DatafeederFileListDummy(nc7.data.DataFeederFileList):
    register_name = "dummy_data_feeder_fl"
    file_list_keys = ['data']


# TODO(oleksandr.vorobiov@audi.de): add tests for dataset mix
class TestRunsBuilder(tf.test.TestCase, parameterized.TestCase):

    def setUp(self):
        reset_register_and_logger()

        register_new_class('dummy_dataset_train',
                           nc7.data.Dataset)
        register_new_class('dummy_dataset_eval',
                           nc7.data.Dataset)
        register_new_class('dummy_callback_train',
                           nc7.coordinator.CoordinatorCallback)
        register_new_class('dummy_plugin_train',
                           nc7.model.ModelPlugin)
        register_new_class('dummy_loss',
                           nc7.model.ModelLoss)
        register_new_class('dummy_pp',
                           nc7.model.ModelPostProcessor)
        register_new_class('dummy_metrics',
                           nc7.model.ModelMetric)
        register_new_class('dummy_summaries',
                           nc7.model.ModelSummary)
        register_new_class('dummy_dataset_fl_train',
                           _DatasetFileListDummy)
        register_new_class('dummy_dataset_fl_eval',
                           _DatasetFileListDummy)
        register_new_class('dummy_file_list',
                           FileListDummy)

        register_new_class('dummy_data_feeder',
                           nc7.data.DataFeeder)
        register_new_class('dummy_callback_infer',
                           nc7.coordinator.CoordinatorCallback)
        register_new_class('dummy_data_feeder_fl',
                           _DatafeederFileListDummy)
        register_new_class('dummy_file_list_infer',
                           FileListDummy)

        register_new_class("dummy_kpi_plugin1",
                           nc7.kpi.KPIPlugin)
        register_new_class("dummy_kpi_plugin2",
                           nc7.kpi.KPIPlugin)
        register_new_class("dummy_kpi_accumulator",
                           nc7.kpi.KPIAccumulator)

    @parameterized.parameters({'dataset_with_file_list': True,
                               "with_kpi": False},
                              {'dataset_with_file_list': False,
                               "with_kpi": True})
    @patch('nucleus7.coordinator.Trainer.build', return_value=-1)
    def test_build_train(self, trainer_build, dataset_with_file_list,
                         with_kpi):
        project_dir = self.get_temp_dir()
        batch_size = 10

        if dataset_with_file_list:
            config_datasets = {
                k: {'class_name': 'dummy_dataset_fl_{}'.format(k)}
                for k in ['train', 'eval']
            }
        else:
            config_datasets = {
                k: {'class_name': 'dummy_dataset_{}'.format(k)}
                for k in ['train', 'eval']
            }

        if dataset_with_file_list:
            for each_mode in config_datasets:
                file_list_config = {
                    'class_name': "dummy_file_list",
                    'file_names': {
                        'data': ['{}_file.ext'.format(each_mode)]
                    }
                }
                config_datasets[each_mode]['file_list'] = file_list_config

        config_callbacks = [{'class_name': 'dummy_callback_train',
                             'name': 'callback1'}]
        config_trainer = {'batch_size': batch_size,
                          'num_epochs': 156,
                          'samples_per_epoch': {'train': 10, 'eval': 20},
                          'optimization_parameters':
                              {'optimizer_name': 'rmsprop',
                               'learning_rate': 1e-3}}
        config_model = {'regularization_l1': 10,
                        'regularization_l2': 20}
        config_plugins = [{'class_name': 'dummy_plugin_train',
                           'name': 'pl_1'}]
        config_losses = [{'class_name': 'dummy_loss',
                          'name': 'l_1'}]
        config_postprocessors = [{'class_name': 'dummy_pp',
                                  'name': 'pp_1'}]
        config_metrics = [{'class_name': 'dummy_metrics',
                           'name': 'm_1'}]
        config_summaries = [{'class_name': 'dummy_summaries',
                             'name': 's_1'}]
        config_kpi = None
        if with_kpi:
            config_kpi = [
                {"class_name": "dummy_kpi_plugin1", "name": "kpi_plugin1"},
                {"class_name": "dummy_kpi_plugin2", "name": "kpi_plugin2"},
                {"class_name": "dummy_kpi_accumulator", "name": "kpi_plugin3"},
            ]
        trainer = runs_builder.build_train(
            project_dir, trainer_config=config_trainer,
            datasets_config=config_datasets,
            callbacks_config=config_callbacks,
            model_config=config_model,
            plugins_config=config_plugins,
            losses_config=config_losses,
            postprocessors_config=config_postprocessors,
            metrics_config=config_metrics,
            kpi_config=config_kpi,
            summaries_config=config_summaries)
        self.assertEqual(trainer_build.call_count, 1)

        config_callback = config_callbacks[0]
        config_callback.pop('class_name', None)
        callback_name = config_callback.pop('name')
        for mode in ['train', 'eval']:
            callbacks = list(zip(*sorted(
                trainer.callbacks_handler[mode].callbacks.items())))[1]
            callback = callbacks[0]
            self.assertTrue(callback.built)
            self.assertEqual(callback_name,
                             callback.name)
            self.assertDictContainsSubset(config_callback,
                                          callback.__dict__)
            if with_kpi and mode == "eval":
                kpi_callback = callbacks[1]
                self.assertTrue(kpi_callback.built)
                self.assertTrue(kpi_callback.evaluator.built)
                for each_item in kpi_callback.evaluator.plugins.values():
                    self.assertTrue(each_item.built)
                for each_item in kpi_callback.evaluator.accumulators.values():
                    self.assertTrue(each_item.built)

                self.assertIsInstance(kpi_callback,
                                      nc7.kpi.kpi_callback.KPIEvaluatorCallback)
                self.assertEqual(2,
                                 len(kpi_callback.evaluator.plugins))
                self.assertEqual(1,
                                 len(kpi_callback.evaluator.accumulators))
                self.assertSetEqual(
                    {"kpi_plugin1", "kpi_plugin2"},
                    {plugin.name
                     for plugin in kpi_callback.evaluator.plugins.values()})
                self.assertSetEqual(
                    {"kpi_plugin3"},
                    {plugin.name
                     for plugin in kpi_callback.evaluator.accumulators.values()}
                )

        for m in ['train', 'eval']:
            config_datasets[m].pop('class_name', None)
            file_list_config = config_datasets[m].pop('file_list', None)
            self.assertDictContainsSubset(
                config_datasets[m], trainer.datasets[m].__dict__)
            self.assertTrue(trainer.datasets[m].built)
            if dataset_with_file_list:
                self.assertTrue(trainer.datasets[m].file_list.built)
                self.assertIsInstance(trainer.datasets[m].file_list,
                                      FileListDummy)
                self.assertDictEqual(file_list_config['file_names'],
                                     trainer.datasets[m].file_list.get())

        model = trainer.model
        for i, conf in enumerate(config_plugins):
            name = conf.pop('name')
            conf.pop('class_name', None)
            self.assertEqual(name, model.plugins[name].name)
            self.assertDictContainsSubset(
                conf, model.plugins[name].__dict__)
            self.assertTrue(model.plugins[name].built)
        for i, conf in enumerate(config_losses):
            name = conf.pop('name')
            conf.pop('class_name', None)
            self.assertEqual(name, model.losses[name].name)
            self.assertDictContainsSubset(
                conf, model.losses[name].__dict__)
            self.assertTrue(model.losses[name].built)
        for i, conf in enumerate(config_postprocessors):
            name = conf.pop('name')
            conf.pop('class_name', None)
            self.assertEqual(name, model.postprocessors[name].name)
            self.assertDictContainsSubset(
                conf, model.postprocessors[name].__dict__)
            self.assertTrue(model.postprocessors[name].built)
        for i, conf in enumerate(config_metrics):
            name = conf.pop('name')
            conf.pop('class_name', None)
            self.assertEqual(name, model.metrics[name].name)
            self.assertDictContainsSubset(
                conf, model.metrics[name].__dict__)
            self.assertTrue(model.metrics[name].built)
        for i, conf in enumerate(config_summaries):
            name = conf.pop('name')
            conf.pop('class_name', None)
            self.assertEqual(name, model.summaries[name].name)
            self.assertDictContainsSubset(
                conf, model.summaries[name].__dict__)
            self.assertTrue(model.summaries[name].built)

    @parameterized.parameters(
        {'datafeeder_with_file_list': True, "provide_batch_size": True,
         'with_kpi': True},
        {'datafeeder_with_file_list': False, "provide_batch_size": True},
        {'datafeeder_with_file_list': False, "provide_batch_size": False},
        {'datafeeder_with_file_list': True, "provide_batch_size": True,
         'use_single_process': True, "prefetch_buffer_size": 10},
        {'datafeeder_with_file_list': True, "provide_batch_size": True,
         'use_single_process': False},
    )
    @patch(
        'nucleus7.coordinator.configs.create_and_validate_inference_load_config',
        autospec=True)
    @patch('nucleus7.coordinator.predictors.predictor_from_load_config',
           return_value=None)
    @patch(
        'nucleus7.coordinator.predictors.represent_predictor_through_nucleotides',
        return_value=[])
    def test_build_inference(self, predictor_repr_fn,
                             get_predictor, create_load_config,
                             datafeeder_with_file_list,
                             provide_batch_size,
                             use_single_process=None,
                             prefetch_buffer_size=None,
                             with_kpi=False):

        project_dir = self.get_temp_dir()
        batch_size = 10
        batch_size_inside_of_inferer_config = 5
        shard_index = 2
        number_of_shards = 3
        graph_fname = 'graph_fname'
        checkpoint_fname = 'checkpoint_fname'
        saved_model_fname = 'saved_model_fname'
        load_config_must = InferenceLoadConfig(saved_model=saved_model_fname,
                                               meta_graph=graph_fname,
                                               checkpoint=checkpoint_fname)
        create_load_config.return_value = load_config_must

        if datafeeder_with_file_list:
            datafeeder_config = {'class_name': 'dummy_data_feeder_fl'}
        else:
            datafeeder_config = {'class_name': 'dummy_data_feeder'}
        config_callbacks = [{'class_name': 'dummy_callback_infer',
                             'name': 'callback1'}]

        if datafeeder_with_file_list:
            file_list_config = {
                'class_name': "dummy_file_list_infer",
                'file_names': {
                    'data': ['infer_file_{}.ext'.format(i)
                             for i in range(number_of_shards)]
                }
            }
            datafeeder_config['file_list'] = file_list_config

        # due to shard of the file names
        file_names_must = {"data": ["infer_file_2.ext"]}

        inferer_config = {
            "run_config": {"batch_size": batch_size_inside_of_inferer_config,
                           "prefetch_buffer_size": 2}}

        config_kpi = None
        if with_kpi:
            config_kpi = [
                {"class_name": "dummy_kpi_plugin1", "name": "kpi_plugin1"},
                {"class_name": "dummy_kpi_plugin2", "name": "kpi_plugin2"},
                {"class_name": "dummy_kpi_accumulator", "name": "kpi_plugin3"},
            ]

        inferer = runs_builder.build_infer(
            project_dir=project_dir,
            batch_size=batch_size if provide_batch_size else None,
            inferer_config=inferer_config,
            kpi_config=config_kpi,
            checkpoint=checkpoint_fname,
            saved_model=saved_model_fname,
            datafeeder_config=datafeeder_config,
            callbacks_config=config_callbacks,
            shard_index=shard_index,
            number_of_shards=number_of_shards,
            use_single_process=use_single_process,
            prefetch_buffer_size=prefetch_buffer_size
        )

        batch_size_must = (batch_size if provide_batch_size
                           else batch_size_inside_of_inferer_config)
        self.assertEqual(batch_size_must, inferer.run_config.batch_size)
        self.assertTupleEqual(load_config_must, inferer.load_config)

        del datafeeder_config['class_name']
        file_list_config = datafeeder_config.pop('file_list', None)
        self.assertDictContainsSubset(datafeeder_config,
                                      inferer.data_feeder.__dict__)
        self.assertTrue(inferer.data_feeder.built)

        config_callback = config_callbacks[0]
        config_callback.pop('class_name', None)
        callback_name = config_callback.pop('name')
        callbacks = list(zip(*sorted(
            inferer.callbacks_handler.callbacks.items())))
        callback = callbacks[1][0]
        self.assertEqual(callback_name,
                         callback.name)
        self.assertTrue(callback.built)
        self.assertDictContainsSubset(config_callback,
                                      callback.__dict__)

        if with_kpi:
            kpi_callback = callbacks[1][1]
            self.assertTrue(kpi_callback.built)
            self.assertIsInstance(kpi_callback.evaluator.dna_helix,
                                  DNAHelix)
            self.assertTrue(kpi_callback.evaluator.built)
            for each_item in kpi_callback.evaluator.plugins.values():
                self.assertTrue(each_item.built)
            for each_item in kpi_callback.evaluator.accumulators.values():
                self.assertTrue(each_item.built)
            self.assertIsInstance(kpi_callback,
                                  nc7.kpi.kpi_callback.KPIEvaluatorCallback)
            self.assertEqual(2,
                             len(kpi_callback.evaluator.plugins))
            self.assertEqual(1,
                             len(kpi_callback.evaluator.accumulators))
            self.assertSetEqual(
                {"kpi_plugin1", "kpi_plugin2"},
                {plugin.name
                 for plugin in kpi_callback.evaluator.plugins.values()})
            self.assertSetEqual(
                {"kpi_plugin3"},
                {plugin.name
                 for plugin in kpi_callback.evaluator.accumulators.values()}
            )

        self.assertTrue(inferer.built)
        self.assertEqual(1, get_predictor.call_count)
        if datafeeder_with_file_list:
            self.assertIsInstance(inferer.data_feeder.file_list,
                                  nc7.data.FileList)
            self.assertTrue(inferer.data_feeder.file_list.built)
            self.assertDictEqual(file_names_must,
                                 inferer.data_feeder.file_list.get())
        if prefetch_buffer_size is not None:
            self.assertEqual(prefetch_buffer_size,
                             inferer.run_config.prefetch_buffer_size)
        else:
            self.assertEqual(2,
                             inferer.run_config.prefetch_buffer_size)
        if use_single_process is not None:
            self.assertEqual(not use_single_process,
                             inferer.run_config.use_multiprocessing)
        else:
            self.assertEqual(True,
                             inferer.run_config.use_multiprocessing)
