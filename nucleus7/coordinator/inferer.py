# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Interface for Inference coordinator
"""

from collections import namedtuple
import logging
import math
import multiprocessing as mp
import time
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Union

import tensorflow as tf
# pylint: disable=no-name-in-module
# is not part of the tensorflow API, but is needed here
# pylint: disable=unused-import
# is used for type declaration
from tensorflow.contrib.predictor.predictor import Predictor

from nucleus7.coordinator import configs as coord_configs
from nucleus7.coordinator import predictors
from nucleus7.coordinator.callbacks_handler import CallbacksHandler
from nucleus7.coordinator.coordinator import Coordinator
from nucleus7.core import project
from nucleus7.core import project_artifacts
from nucleus7.core import project_serializer
from nucleus7.core.dna_helix import DNAHelix
from nucleus7.data.data_feeder import DataFeeder
from nucleus7.utils import mlflow_utils
from nucleus7.utils import nest_utils
from nucleus7.utils import nucleotide_utils
from nucleus7.utils import object_utils

# pylint: enable=unused-import,no-name-in-module


_QUEUE_COMPLETED = "QUEUE_COMPLETED"
_InfererQueues = namedtuple("InfererQueues", ("data_feeder", "results"))
_InfererProcesses = namedtuple("InfererProcesses",
                               ("data_feeder", "callbacks_handler"))


# pylint: disable=too-many-instance-attributes
# attributes cannot be combined or extracted further
class Inferer(Coordinator):
    """
    Class for coordination of inference

    Parameters
    ----------
    project_dir
        project directory
    run_config
        run configuration for inferer
    load_config
        inferer load configuration
    tensorrt_config
        tensorrt config to use for inference
    data_feeder
        used for feeding the data to session using feed_dict
    model_incoming_keys_mapping
        mapping of keys to use between data feeder and the model; if not set,
        input keys from the model, e.g. collection names inside of
        CollectionNames.INPUTS collection should match desired data feeder
        keys
    model_parameters
        parameters of the model, that wre defined as default placeholders in
        nested view, e.g. {'nucleotide_name': {'parameter1': 100}}
    callbacks_handler
        handler with callbacks for inference
    session_config
        session configuration passed to `tf.Session(config=session_config)`
    use_model
        if the inferer should use a tensorflow model as predictor after
        data feeder and before callbacks; otherwise only callbacks and
        data feeder will be used
    project_type
        type of the inference project; one of [infer, kpi, data_extraction]
    project_additional_kwargs
        additional kwargs to pass to project generation method

    Generates following project structure, where saved_model_tag is a last
    directory of saved_model or checkpoint file name, e.g. in path
    `model.ckpt-100` subdir will be model-ckpt-100:

        - {project_dir}/infer_configs/{saved_model_tag}
        - {project_dir}/infer_results/{saved_model_tag}

    Attributes
    ----------
    register_name_scope
        name scope for register and for the config logger
    exclude_args_from_log
        fields that will not be included to the config logger
    _queues
        queues between data_feeder and model and model and callbacks_handler
    _processes
        data feeder process and callbacks handler process
    _predictor
        predictor to use
    _iteration_number
        iteration number
    _number_of_iterations
        number of iteration

    See Also
    --------
    :obj:`nucleus7.coordinator.configs.InferenceRunConfig`
    :obj:`nucleus7.coordinator.configs.InferenceLoadConfig`

    """
    register_name_scope = "inferer"
    exclude_from_register = True
    exclude_from_log = False
    exclude_args_from_log = ["data_feeder", "callbacks_handler"]

    def __init__(
            self, project_dir: str, *,
            data_feeder: DataFeeder,
            run_config: coord_configs.InferenceRunConfig,
            load_config: Optional[coord_configs.InferenceLoadConfig] = None,
            tensorrt_config: Optional[coord_configs.TensorrtConfig] = None,
            model_incoming_keys_mapping: Optional[dict] = None,
            model_parameters: Optional[dict] = None,
            callbacks_handler: Optional[CallbacksHandler] = None,
            session_config: Optional[dict] = None,
            use_model: Optional[bool] = True,
            project_type: Optional[str] = "infer",
            project_additional_kwargs: Optional[dict] = None):
        super(Inferer, self).__init__(project_dir, run_config,
                                      callbacks_handler=callbacks_handler,
                                      session_config=session_config)
        if use_model and not load_config:
            raise ValueError(
                "load_config must be provided if you want to use a model")
        if not use_model and load_config:
            load_config = None
        if project_type not in ["infer", "kpi", "data_extraction"]:
            raise ValueError("project_type must be one of "
                             "[infer, kpi, data_extraction]! "
                             "(provided: {})".format(project_type))

        self.load_config = load_config
        self.data_feeder = data_feeder
        self.model_incoming_keys_mapping = model_incoming_keys_mapping
        self.model_parameters = model_parameters or {}
        self.tensorrt_config = tensorrt_config
        self.use_model = use_model
        self.project_type = project_type
        self.project_additional_kwargs = project_additional_kwargs or {}
        self._predictor_dna_helix = None  # type: DNAHelix
        self._predictor = None  # type: Optional[Predictor]
        self._queues = None  # type: _InfererQueues
        self._processes = None  # type: _InfererProcesses

    @property
    def dna_helices(self) -> Optional[Dict[str, DNAHelix]]:
        if not self.built:
            return None

        project_dna_helix = (self._predictor_dna_helix
                             + self.callbacks_handler.dna_helix)
        dna_helices = {tf.estimator.ModeKeys.PREDICT: project_dna_helix}
        dna_helices_kpi = self.callbacks_handler.kpi_evaluators_dna_helices
        if dna_helices_kpi:
            dna_helices.update(dna_helices_kpi)
        if hasattr(self.data_feeder, "dna_helix"):
            dna_helices["dataset"] = self.data_feeder.dna_helix
        return dna_helices

    @property
    def predictor_dna_helix(self) -> DNAHelix:
        """
        Returns
        -------
        predictor_dna_helix
            dna helix with data feeder and predictor nucleotides
        """
        return self._predictor_dna_helix

    def build(self):
        super().build()
        project.create_or_get_active_project(
            project_type=self.project_type,
            project_dir=self.project_dir,
            continue_last=self.run_config.continue_last,
            **self.project_additional_kwargs)
        self.project_dirs = project.get_active_project_dirs()
        if self.use_model:
            self._predictor = self.get_predictor()
        else:
            self._predictor = None
        self._build_predictor_dna()
        self._build_callbacks_handler_dna()
        if self.data_feeder.samples_n == -1:
            number_iterations_per_epoch = -1
        else:
            number_iterations_per_epoch = math.ceil(
                self.data_feeder.samples_n / self.run_config.batch_size)
        self.set_callback_properties(tf.estimator.ModeKeys.PREDICT,
                                     self.project_dirs.callbacks,
                                     number_iterations_per_epoch)
        return self

    @object_utils.assert_is_built
    @mlflow_utils.create_mlflow_experiment_and_start_run
    @project_serializer.serialize_run_configs(
        "infer_run",
        single_config_names=["inferer", "data_feeder"],
        serializers_to_use=[project_serializer.RunConfigSerializer,
                            project_serializer.MlflowConfigSerializer])
    @project_artifacts.serialize_project_artifacts
    @mlflow_utils.log_project_artifacts_to_mlflow
    def run(self):
        """
        Run the inference
        """
        logger = logging.getLogger(__name__)
        logger.info("START %s", self.__class__.__name__.upper())
        time_start = time.time()
        self._run()
        time_end = time.time() - time_start
        logger.info("Completed in {:0.3f} s".format(time_end))
        logger.info("DONE!!!")

    def get_predictor(self) -> Predictor:
        """
        Initialize predictor from load config

        Returns
        -------
        predictor
            predictor from load_config
        """
        session_config = tf.ConfigProto(**self.session_config)
        return predictors.predictor_from_load_config(
            load_config=self.load_config,
            tensorrt_config=self.tensorrt_config,
            session_config=session_config,
            postprocessors_to_use=self.run_config.postprocessors_to_use,
            model_parameters=self.model_parameters,
        )

    @mlflow_utils.log_nucleotide_exec_time_to_mlflow(
        method_name="read_data_batch")
    def read_data_batch(self, last_inputs: Optional[dict]
                        ) -> Optional[Tuple[dict, dict, bool]]:
        """
        Read the batch and add it to the queue

        Parameters
        ----------
        last_inputs
            last input that was generated; it is needed to understand if this
            is the last iteration
        """
        is_last_iteration = False
        try:
            if last_inputs is None:
                inputs = self.data_feeder(self.run_config.batch_size)
            else:
                inputs = last_inputs
            try:
                last_inputs = self.data_feeder(self.run_config.batch_size)
            except StopIteration:
                is_last_iteration = True

            return inputs, last_inputs, is_last_iteration
        except StopIteration:
            return None

    @mlflow_utils.log_nucleotide_exec_time_to_mlflow(
        method_name="predict_batch")
    def predict_batch(self, inputs: Union[dict, list]) -> Tuple[dict, float]:
        """
        Make predictions given inputs

        Parameters
        ----------
        inputs
            inputs to the network

        Returns
        -------
        predictions
            predictions of the network
        predict_exec_time
            execution time of network prediction
        """
        time_start_predict = time.time()
        if not isinstance(inputs, list):
            inputs = [inputs]
        list_of_predictions = []
        for each_input in inputs:
            if self.model_incoming_keys_mapping is not None:
                each_input = nucleotide_utils.remap_and_collapse_inputs(
                    [each_input], [self.model_incoming_keys_mapping])
            each_input_flatten = nest_utils.flatten_nested_struct(each_input)
            current_prediction_flatten = predictors.predict_using_predictor(
                predictor=self._predictor, inputs=each_input_flatten,
                model_parameters=self.model_parameters)
            list_of_predictions.append(current_prediction_flatten)
        predictions_flatten = nucleotide_utils.collapse_inputs(
            list_of_predictions)
        predictions = nest_utils.unflatten_dict_to_nested(predictions_flatten)
        predict_exec_time = time.time() - time_start_predict
        return predictions, predict_exec_time

    @mlflow_utils.log_nucleotide_exec_time_to_mlflow(
        method_name="run_callbacks_handler")
    def run_callbacks_handler_on_batch(
            self, inputs: dict, predictions: dict, predict_exec_time: float,
            iteration_number: int, is_last_iteration: bool):
        """
        Run callbacks handler on batch of predictions and inputs

        Parameters
        ----------
        inputs
            inputs from data_feeder
        predictions
            predictions from network
        predict_exec_time
            execution time of network
        iteration_number
            iteration number
        is_last_iteration
            flag that indicates that this iteration is last one
        """
        self.callbacks_handler.on_iteration_start()
        if isinstance(inputs, list):
            inputs = nucleotide_utils.collapse_inputs(inputs)
        predictions['dataset'] = inputs
        iteration_info = coord_configs.RunIterationInfo(
            epoch_number=1, iteration_number=iteration_number,
            execution_time=predict_exec_time,
            is_last_iteration=is_last_iteration
        )
        self.callbacks_handler.iteration_info = iteration_info
        callbacks_handler_inputs = self.callbacks_handler.filter_inputs(
            inputs=predictions)
        self.callbacks_handler(**callbacks_handler_inputs)

    def _run(self):
        if self.run_config.use_multiprocessing:
            self._run_multiprocessing()
        else:
            self._run_single_process()

    def _run_single_process(self):
        last_inputs = None
        iteration_number = 1
        self.callbacks_handler.begin()
        while True:
            batch = self.read_data_batch(last_inputs)
            if batch is None:
                break
            inputs, last_inputs, is_last_iteration = batch
            if self.use_model:
                predictions, predict_exec_time = self.predict_batch(inputs)
            else:
                predictions, predict_exec_time = {}, -1.0
            self.run_callbacks_handler_on_batch(
                inputs, predictions, predict_exec_time,
                iteration_number, is_last_iteration)
            iteration_number += 1
            if is_last_iteration:
                break
        self.callbacks_handler.end()

    def _run_multiprocessing(self):
        self._create_queues()
        self._create_processes()
        self._run_processes()
        self._close_queues()
        self._terminate_processes()

    def _run_processes(self):
        self._processes.data_feeder.start()
        self._processes.callbacks_handler.start()
        if self.use_model:
            self._predict_batch_in_separate_process()
        self._processes.data_feeder.join()
        self._processes.callbacks_handler.join()

    def _create_queues(self):
        data_feeder_queue = mp.Queue(self.run_config.prefetch_buffer_size)
        if self.use_model:
            results_queue = mp.Queue(self.run_config.prefetch_buffer_size)
        else:
            results_queue = None
        self._queues = _InfererQueues(data_feeder_queue, results_queue)

    def _create_processes(self):
        data_feeder_process = mp.Process(
            target=self._prefetch_data_in_separate_process, daemon=True)
        callbacks_handler_process = mp.Process(
            target=self._run_callbacks_handler_on_batch_in_separate_process,
            daemon=True)
        self._processes = _InfererProcesses(
            data_feeder_process, callbacks_handler_process)

    def _close_queues(self):
        self._queues.data_feeder.close()
        if self._queues.results is not None:
            self._queues.results.close()

    def _terminate_processes(self):
        self._processes.data_feeder.terminate()
        self._processes.callbacks_handler.terminate()

    def _prefetch_data_in_separate_process(self):
        last_inputs = None
        iteration_number = 1
        while True:
            inputs_queue = self.read_data_batch(last_inputs)
            if inputs_queue is None:
                break

            inputs, last_inputs, is_last_iteration = inputs_queue
            queue_data = (
                (inputs, iteration_number, is_last_iteration)
                if self.use_model
                else (inputs, {}, -1.0, iteration_number, is_last_iteration))
            self._queues.data_feeder.put(queue_data)
            if is_last_iteration:
                break
            iteration_number += 1
        self._queues.data_feeder.put(_QUEUE_COMPLETED)

    def _predict_batch_in_separate_process(self):
        while True:
            inputs_deque = self._queues.data_feeder.get()
            if inputs_deque == _QUEUE_COMPLETED:
                self._queues.results.put(_QUEUE_COMPLETED)
                break
            inputs, iteration_number, is_last_iteration = inputs_deque
            predictions, predict_exec_time = self.predict_batch(inputs)
            inputs_for_callbacks = (inputs, predictions, predict_exec_time,
                                    iteration_number, is_last_iteration)
            self._queues.results.put(inputs_for_callbacks)

    def _run_callbacks_handler_on_batch_in_separate_process(self):
        self.callbacks_handler.begin()
        while True:
            if self.use_model:
                inputs_deque = self._queues.results.get()
            else:
                inputs_deque = self._queues.data_feeder.get()
            if inputs_deque == _QUEUE_COMPLETED:
                break

            self.run_callbacks_handler_on_batch(*inputs_deque)
        self.callbacks_handler.end()

    def _build_predictor_dna(self):
        if not self.use_model:
            self._predictor_dna_helix = DNAHelix(nucleotides=[]).build()
            return

        incoming_nucleotides = [self.data_feeder]
        predictor_nucleotides = (
            predictors.represent_predictor_through_nucleotides(
                self._predictor,
                incoming_keys_mapping=self.model_incoming_keys_mapping))
        self._predictor_dna_helix = DNAHelix(
            nucleotides=predictor_nucleotides,
            incoming_nucleotides=incoming_nucleotides).build()

    def _build_callbacks_handler_dna(self):
        incoming_nucleotides = ([self.data_feeder]
                                + self._predictor_dna_helix.nucleotides)
        self.callbacks_handler.build_dna(
            incoming_nucleotides=incoming_nucleotides)
