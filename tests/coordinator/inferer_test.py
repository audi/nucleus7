# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

import itertools
import os
import threading
from unittest.mock import MagicMock
from unittest.mock import call as mock_call
from unittest.mock import patch

from absl.testing import parameterized
import numpy as np
import tensorflow as tf
from tensorflow.contrib.predictor.predictor import Predictor

from nucleus7.coordinator import configs as cconfigs
from nucleus7.coordinator import predictors
from nucleus7.coordinator.callbacks_handler import CallbacksHandler
from nucleus7.coordinator.inferer import Inferer
from nucleus7.data import DataFeeder
from nucleus7.test_utils.test_utils import reset_register_and_logger
from nucleus7.utils import nest_utils


class _MockProcess(threading.Thread):
    def terminate(self) -> None:
        pass


class TestInferer(parameterized.TestCase, tf.test.TestCase):

    def setUp(self):
        reset_register_and_logger()
        tf.reset_default_graph()
        np.random.seed(6546)
        self.batch_size = 8
        self.predictor_inputs_np = {
            "data1": np.random.randn(self.batch_size, 2).astype(np.float32)}
        self.predictor_outputs_np = {
            'node1': {
                'image': np.random.randn(self.batch_size, 2).astype(
                    np.float32),
                'data': np.random.randn(self.batch_size, 1).astype(
                    np.float32)
            },
            'node2': {
                'temp': np.random.randn(self.batch_size).astype(np.float32)
            }
        }
        self.inputs_dataset = {'data1': 10,
                               'data2': 20}

    @parameterized.parameters({'feed_list_len': 0},
                              {'feed_list_len': 3})
    @patch('nucleus7.coordinator.predictors.predictor_from_load_config')
    def test_build(self, get_predictor_fn, feed_list_len):
        project_dir = self.get_temp_dir()
        data_feeder = self._get_mock_data_feeder(feed_list_len)
        callbacks_handler = self._get_callbacks_handler()
        load_config = self._get_load_config(project_dir)
        run_config = self._get_run_config()

        session = self.test_session()
        predictor = self._get_predictor_mock(session)
        get_predictor_fn.return_value = predictor

        inferer = Inferer(project_dir, data_feeder=data_feeder,
                          run_config=run_config, load_config=load_config,
                          callbacks_handler=callbacks_handler)
        inferer.build()
        callbacks_dir_must = os.path.join(
            project_dir, "inference", "run-1", "results")
        self.assertTrue(os.path.isdir(callbacks_dir_must))

        callbacks_handler = inferer.callbacks_handler
        self.assertEqual(1,
                         inferer.callbacks_handler.build_dna.call_count)

        build_dna_handler_args, build_dna_handler_kwargs = (
            callbacks_handler.build_dna.call_args)
        self.assertEqual(0, len(build_dna_handler_args))
        self.assertSetEqual({'incoming_nucleotides'},
                            set(build_dna_handler_kwargs))
        incoming_nucleotides = build_dna_handler_kwargs['incoming_nucleotides']

        predictor_nucleotides_must = (
            predictors.represent_predictor_through_nucleotides(predictor))
        self.assertEqual(len(predictor_nucleotides_must) + 1,
                         len(incoming_nucleotides))
        self.assertIn(inferer.data_feeder,
                      incoming_nucleotides)
        incoming_nucleotides_without_data_feeder = [
            nucleotide for nucleotide in incoming_nucleotides
            if nucleotide is not inferer.data_feeder
        ]

        for incoming_nucleotide_must, incoming_nucleotide in zip(
                predictor_nucleotides_must,
                incoming_nucleotides_without_data_feeder):
            self.assertEqual(incoming_nucleotide_must.name,
                             incoming_nucleotide.name)
            self.assertSetEqual(set(incoming_nucleotide_must.generated_keys),
                                set(incoming_nucleotide.generated_keys))
            self.assertSetEqual(set(incoming_nucleotide_must.inbound_nodes),
                                set(incoming_nucleotide.inbound_nodes))

        self.assertEqual(tf.estimator.ModeKeys.PREDICT,
                         callbacks_handler.mode)
        self.assertEqual(inferer.project_dirs.callbacks,
                         callbacks_handler.log_dir)
        self.assertEqual(-1,
                         callbacks_handler.number_iterations_per_epoch)

    @parameterized.parameters({"number_of_iterations": 0},
                              {"number_of_iterations": 1},
                              {"number_of_iterations": 3})
    def test_read_data_batch(self, number_of_iterations):
        project_dir = self.get_temp_dir()
        data_feeder = self._get_mock_data_feeder(0, number_of_iterations)
        run_config = self._get_run_config()
        load_config = self._get_load_config(project_dir)
        inferer = Inferer(project_dir, data_feeder=data_feeder,
                          run_config=run_config, load_config=load_config,
                          callbacks_handler=None)
        predictor = self._get_predictor_mock(None)
        inferer.get_predictor = MagicMock(return_value=predictor)
        inferer.build()
        batch_size = inferer.run_config.batch_size

        res = inferer.read_data_batch(None)
        if number_of_iterations == 0:
            self.assertIsNone(res)
            data_feeder.get_batch.assert_called_once_with(batch_size)
            return

        inputs, last_inputs, is_last_iteration = res
        self.assertDictEqual({"data1": 0, "data2": 0},
                             inputs)
        if number_of_iterations == 1:
            self.assertTrue(is_last_iteration)
            self.assertIsNone(last_inputs)
            res = inferer.read_data_batch(last_inputs)
            self.assertIsNone(res)
            data_feeder.get_batch.assert_has_calls(
                [mock_call(batch_size), mock_call(batch_size)])
            return

        last_i = number_of_iterations - 2
        for i in range(number_of_iterations + 2):
            res = inferer.read_data_batch(last_inputs)
            inputs, last_inputs, is_last_iteration = res
            if i >= last_i:
                self.assertTrue(is_last_iteration)
            else:
                self.assertFalse(is_last_iteration)
            if i <= last_i:
                self.assertDictEqual({"data1": i + 1, "data2": i + 1},
                                     inputs)
            else:
                self.assertDictEqual({"data1": number_of_iterations - 1,
                                      "data2": number_of_iterations - 1},
                                     inputs)
                self.assertDictEqual(inputs,
                                     last_inputs)

        data_feeder.get_batch.assert_has_calls(
            [mock_call(batch_size)] * (number_of_iterations + 1))

    @parameterized.parameters(
        {'feed_list_len': 0},
        {'feed_list_len': 1},
        {'feed_list_len': 3},
        {'feed_list_len': 3,
         'model_parameters': {"nucleotide1": {"parameter1": 100}}})
    @patch("nucleus7.coordinator.predictors.predict_using_predictor")
    def test_predict_batch(self, predict_using_predictor_fn, feed_list_len,
                           model_parameters=None):
        predictor_results = {"result": 1}
        predict_using_predictor_fn.return_value = {"result": 1}
        project_dir = self.get_temp_dir()
        data_feeder = self._get_mock_data_feeder(0, 1)
        run_config = self._get_run_config()
        load_config = self._get_load_config(project_dir)
        inferer = Inferer(project_dir, data_feeder=data_feeder,
                          run_config=run_config, load_config=load_config,
                          callbacks_handler=None,
                          model_parameters=model_parameters)
        predictor = self._get_predictor_mock(None)
        inferer.get_predictor = MagicMock(return_value=predictor)
        inferer.build()
        inputs = {"data1": 1, "data2": 2}
        if feed_list_len > 0:
            inputs = [
                {"data1": i, "data2": i + 1} for i in range(feed_list_len)]
        res = inferer.predict_batch(inputs)
        self.assertIsInstance(res, tuple)
        self.assertLen(res, 2)
        if feed_list_len > 1:
            results_must = {"result": [1] * feed_list_len}
        else:
            results_must = predictor_results
        self.assertAllEqual(results_must,
                            res[0])
        self.assertIsInstance(res[1], float)
        number_of_predictor_calls_must = feed_list_len and feed_list_len or 1
        self.assertEqual(number_of_predictor_calls_must,
                         predict_using_predictor_fn.call_count)

        model_parameters_in_call = model_parameters or {}
        if feed_list_len == 0:
            predict_using_predictor_fn.assert_called_once_with(
                inputs=inputs, predictor=predictor,
                model_parameters=model_parameters_in_call)
        elif feed_list_len == 1:
            predict_using_predictor_fn.assert_called_once_with(
                inputs=inputs[0], predictor=predictor,
                model_parameters=model_parameters_in_call)
        else:
            predict_using_predictor_fn.assert_has_calls(
                [mock_call(inputs=inp, predictor=predictor,
                           model_parameters=model_parameters_in_call)
                 for inp in inputs])

    @parameterized.parameters({"is_last_iteration": True},
                              {"is_last_iteration": False})
    def test_run_callbacks_handler_on_batch(self, is_last_iteration=True):
        predict_exec_time = 0.1
        iteration_number = 7
        project_dir = self.get_temp_dir()
        data_feeder = self._get_mock_data_feeder(0, 1)
        run_config = self._get_run_config()
        callbacks_handler = self._get_callbacks_handler()
        callbacks_handler.on_iteration_start = MagicMock(return_value=None)
        load_config = self._get_load_config(project_dir)
        inferer = Inferer(project_dir, data_feeder=data_feeder,
                          run_config=run_config, load_config=load_config,
                          callbacks_handler=callbacks_handler)
        predictor = self._get_predictor_mock(None)
        inferer.get_predictor = MagicMock(return_value=predictor)
        inferer.build()
        inputs = self.inputs_dataset
        predictions = {"node1": {"prediction1": 100, "prediction2": 200},
                       "node2": {"prediction3": 1}}
        inferer.run_callbacks_handler_on_batch(
            inputs, predictions, predict_exec_time, iteration_number,
            is_last_iteration)
        inputs_to_call_must = {
            "dataset": {"data1": 10, "data2": 20},
            "node1": {"prediction1": 100, "prediction2": 200}}
        self.assertEqual(is_last_iteration,
                         callbacks_handler.iteration_info.is_last_iteration)
        self.assertEqual(iteration_number,
                         callbacks_handler.iteration_info.iteration_number)
        self.assertEqual(predict_exec_time,
                         callbacks_handler.iteration_info.execution_time)
        self.assertEqual(1,
                         callbacks_handler.iteration_info.epoch_number)
        callbacks_handler.on_iteration_start.assert_called_once_with()
        callbacks_handler.process_gene.assert_called_once_with(
            gene_inputs=inputs_to_call_must, gene_name="callbacks")

    @parameterized.parameters(
        dict(zip(("feed_list_len", "use_multiprocessing", "use_model"),
                 values))
        for values in itertools.product(
            [-1, 0, 2], [True, False], [True, False]))
    @patch("multiprocessing.Process", autospec=True)
    def test_run(self, mp_process_mock, feed_list_len, use_multiprocessing,
                 use_model):
        def _create_mp(*args, **kwargs):
            return _MockProcess(*args, **kwargs)

        predict_exec_time = 0.1
        number_of_iterations = 11

        def _predict_batch(inputs):
            if isinstance(inputs, list):
                inputs = inputs[0]
            return {k + "_out": v for k, v in inputs.items()}, predict_exec_time

        mp_process_mock.side_effect = _create_mp
        project_dir = self.get_temp_dir()
        data_feeder = self._get_mock_data_feeder(
            feed_list_len, number_of_iterations)
        callbacks_handler = self._get_callbacks_handler()
        callbacks_handler.begin = MagicMock(return_value=None)
        callbacks_handler.end = MagicMock(return_value=None)
        run_config = self._get_run_config()
        run_config = run_config._replace(
            use_multiprocessing=use_multiprocessing)

        load_config = self._get_load_config(project_dir)
        inferer = Inferer(project_dir, data_feeder=data_feeder,
                          run_config=run_config, load_config=load_config,
                          callbacks_handler=callbacks_handler,
                          use_model=use_model)
        predictor = self._get_predictor_mock(None)
        inferer.get_predictor = MagicMock(return_value=predictor)
        inferer.read_data_batch = MagicMock(wraps=inferer.read_data_batch)
        inferer.predict_batch = MagicMock(side_effect=_predict_batch)
        inferer.run_callbacks_handler_on_batch = MagicMock(return_value=None)
        inferer.build()
        inferer.run()

        if feed_list_len < 0:
            inferer.read_data_batch.assert_called_once_with(None)
            inferer.predict_batch.assert_not_called()
            inferer.run_callbacks_handler_on_batch.assert_not_called()
            return

        predict_batch_num_calls_must = number_of_iterations if use_model else 0

        self.assertEqual(1,
                         callbacks_handler.begin.call_count)
        self.assertEqual(1,
                         callbacks_handler.end.call_count)
        self.assertEqual(number_of_iterations,
                         inferer.read_data_batch.call_count)
        self.assertEqual(predict_batch_num_calls_must,
                         inferer.predict_batch.call_count)
        self.assertEqual(number_of_iterations,
                         inferer.run_callbacks_handler_on_batch.call_count)

        inputs_predict_batch_must = [
            {k: i for k in self.inputs_dataset}
            for i in range(number_of_iterations)
        ]
        predictions_batch_must = [
            {k + "_out": v for k, v in inputs.items()}
            for inputs in inputs_predict_batch_must
        ]
        if feed_list_len > 0:
            inputs_predict_batch_must = [
                [i] * feed_list_len for i in inputs_predict_batch_must]
        if not use_model:
            predictions_batch_must = [{} for _ in range(number_of_iterations)]

        predict_exec_time_must = predict_exec_time if use_model else -1.0
        inputs_run_callbacks_handler_on_batch_must = [
            (inputs, predictions, predict_exec_time_must, iter_number + 1,
             (iter_number == number_of_iterations - 1))
            for iter_number, (inputs, predictions) in enumerate(
                zip(inputs_predict_batch_must, predictions_batch_must))
        ]

        if use_model:
            inferer.predict_batch.assert_has_calls(
                [mock_call(i) for i in inputs_predict_batch_must])
        else:
            inferer.predict_batch.assert_not_called()
        inferer.run_callbacks_handler_on_batch.assert_has_calls(
            [mock_call(*i) for i in inputs_run_callbacks_handler_on_batch_must])

    @staticmethod
    def _get_callbacks_handler():
        class CallbacksHandlerMock(CallbacksHandler):
            def __init__(self, **kwargs):
                super(CallbacksHandlerMock, self).__init__(**kwargs)
                self._inbound_nodes = None

            @property
            def inbound_nodes(self):
                return self._inbound_nodes

        callbacks_handler = CallbacksHandlerMock(callbacks=[]).build()
        callbacks_handler._inbound_nodes = ['dataset', 'node1']
        callbacks_handler.build_dna = MagicMock(return_value=None)
        callbacks_handler.process_gene = MagicMock(return_value=None)
        return callbacks_handler

    @staticmethod
    def _get_load_config(project_dir):
        load_config = cconfigs.InferenceLoadConfig(
            saved_model=project_dir, meta_graph=None, checkpoint=None)
        return load_config

    def _get_run_config(self):
        run_config = cconfigs.InferenceRunConfig(batch_size=self.batch_size)
        return run_config

    def _get_mock_data_feeder(self, feed_list_len, num_iterations=0):
        def _generator():
            for i in range(num_iterations):
                yield i

        generator = _generator()
        inputs_dataset = self.inputs_dataset

        def _get_batch(batch_size):
            i = next(generator)
            if feed_list_len < 0:
                raise StopIteration()
            res_values = {k: i for k in inputs_dataset}
            if feed_list_len > 0:
                res_values = [res_values for _ in range(feed_list_len)]

            return res_values

        data_feeder = DataFeeder()
        data_feeder.generated_keys = list(self.inputs_dataset)
        data_feeder.get_batch = MagicMock(side_effect=_get_batch)
        data_feeder._built = True
        return data_feeder

    def _get_predictor_mock(self, session):
        class _MockPredictor(Predictor):
            def __init__(self_, model_inputs, model_outputs, session):
                self_._feed_tensors = model_inputs
                self_._fetch_tensors = model_outputs
                self_._session = session

        inputs_tf = {k: tf.placeholder(tf.float32, v.shape)
                     for k, v in self.predictor_inputs_np.items()}

        predictor_outputs_np_flatten = nest_utils.flatten_nested_struct(
            self.predictor_outputs_np)
        outputs_tf_flatten = {k + '_out': tf.identity(v)
                              for k, v in predictor_outputs_np_flatten.items()}
        predictor = _MockPredictor(inputs_tf, outputs_tf_flatten,
                                   session)
        return predictor
