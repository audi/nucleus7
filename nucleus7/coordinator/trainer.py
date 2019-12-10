# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Interface for Trainer coordinator
"""

from functools import partial
import logging
import os
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import tensorflow as tf

from nucleus7.coordinator import configs as coord_configs
from nucleus7.coordinator import exporters
from nucleus7.coordinator import session_run_hooks
from nucleus7.coordinator.callbacks_handler import CallbacksHandler
from nucleus7.coordinator.coordinator import Coordinator
from nucleus7.core import project
from nucleus7.core import project_artifacts
from nucleus7.core import project_serializer
from nucleus7.core.dna_helix import DNAHelix
from nucleus7.core.nucleotide import Nucleotide
from nucleus7.data.dataset import Dataset
from nucleus7.model.fields import ScopeNames
from nucleus7.model.model import Model
from nucleus7.model.model_handler import ModelHandler
from nucleus7.optimization import configs as opt_configs
from nucleus7.utils import mlflow_utils
from nucleus7.utils import object_utils
from nucleus7.utils import run_utils


# pylint: disable=too-many-instance-attributes
# is needed to make the trainer more generic
class Trainer(Coordinator):
    """
    Class for coordination of neural networks training with evaluation

    Parameters
    ----------
    model
        instance of model
    project_dir
        project directory, where results will be stored
    run_config
        run configuration for Trainer
    save_config
        save configuration for Trainer
    datasets
        dict with objects of type :obj:`Dataset` and ['train', 'eval'] as keys
    optimization_config
        global optimization configuration
    callbacks_handler_train
        callbacks handler with callbacks to use during training;
        these callbacks are converted to :obj:`tf.train.SessionRunHooks` and
        executed as chief_only_hooks
    callbacks_handler_eval
        callbacks handler with callbacks to use during evaluation;
        these callbacks are converted to :obj:`tf.train.SessionRunHooks` and
        executed as chief_only_hooks
    session_config
        dict with session configuration, which will be passed to the
        monitored training session as
        session_config=tf.ConfigProto(session_config)

    Attributes
    ----------
    register_name_scope
        name scope for register and for the config logger
    exclude_args_from_log
        fields that will not be included to the config logger
    _estimator
        estimator that will be used for training and evaluation

    Generates following project structure:

        - {project_dir}/checkpoints/ : checkpoints
        - {project_dir}/training_callbacks/ : results of training callbacks
        - {project_dir}/training_configs/ : configs for training with runs \
            subfolders
        - {project_dir}/training_summaries/ : training summaries, eval and train
        - {project_dir}/saved_models/ : saved models in timestamp subfolders \
            and inference_graph.meta together with input_output_names.json

    References
    ----------
    mixed precision
        https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html

    """
    register_name_scope = "trainer"
    exclude_from_register = True
    exclude_from_log = False
    exclude_args_from_log = ["model", "datasets", "callbacks_handler_train",
                             "callbacks_handler_eval"]

    def __init__(self, *,
                 project_dir: str,
                 model: Model,
                 datasets: Dict[str, Dataset],
                 run_config: coord_configs.TrainerRunConfig,
                 save_config: coord_configs.TrainerSaveConfig,
                 optimization_config: opt_configs.OptimizationConfig,
                 callbacks_handler_train: Union[CallbacksHandler, None] = None,
                 callbacks_handler_eval: Union[CallbacksHandler, None] = None,
                 session_config=None):
        callbacks_handlers = {
            tf.estimator.ModeKeys.TRAIN: callbacks_handler_train,
            tf.estimator.ModeKeys.EVAL: callbacks_handler_eval
        }
        super().__init__(project_dir, run_config,
                         callbacks_handler=callbacks_handlers,
                         session_config=session_config)
        self.save_config = save_config
        self.model = model
        self.optimization_config = optimization_config
        self.datasets = datasets
        self._model_handler = None  # type: ModelHandler
        self._server_input_receiver_fn_builder = None
        self._estimator = None
        self._estimator_train_spec = None
        self._estimator_eval_spec = None

    @property
    def dna_helices(self) -> Optional[Dict[str, DNAHelix]]:
        if not self.built:
            return None

        dna_helices = {}
        for each_mode, each_callback_handler in self.callbacks_handler.items():
            mode_helix = self.model.dna_helix + each_callback_handler.dna_helix
            dna_helices[each_mode] = mode_helix
            dna_helices_kpi = each_callback_handler.kpi_evaluators_dna_helices
            if dna_helices_kpi:
                dna_helices.update(dna_helices_kpi)
            dataset = self.datasets.get(each_mode)
            if dataset and getattr(dataset, "dna_helix", None) is not None:
                dna_helices["dataset_{}".format(each_mode)] = dataset.dna_helix
        return dna_helices

    @property
    def estimator(self) -> Optional[tf.estimator.Estimator]:
        """
        Estimator used by trainer

        Returns
        -------
        estimator
            estimator
        """
        return self._estimator

    @property
    def estimator_train_spec(self) -> Optional[tf.estimator.TrainSpec]:
        """
        Estimator train spec

        Returns
        -------
        estimator_train_spec
            train spec
        """
        return self._estimator_train_spec

    @property
    def estimator_eval_spec(self) -> Optional[tf.estimator.EvalSpec]:
        """
        Estimator eval spec

        Returns
        -------
        estimator_eval_spec
            eval spec
        """
        return self._estimator_eval_spec

    def get_all_nucleotides(self, mode: str) -> Dict[str, Nucleotide]:
        """
        All nucleotides including callbacks, datasets and model

        Parameters
        ----------
        mode
            mode to get nucleotides

        Returns
        -------
        all_nucleotides_for_mode
            all nucleotides for mode
        """
        all_nucleotides = {}
        if mode in self.datasets:
            dataset_nucleotide = self.datasets[mode]
            all_nucleotides[dataset_nucleotide.name] = dataset_nucleotide
        if self.model.all_nucleotides is not None:
            all_nucleotides.update(self.model.all_nucleotides)
        callbacks_handler = self.callbacks_handler[mode]
        if callbacks_handler.all_nucleotides is not None:
            all_nucleotides.update(callbacks_handler.all_nucleotides)
        return all_nucleotides

    def build(self):
        """
        Build trainer:

            * create project directories
            * create builder for model_fn to pass to the estimator
            * create builder for server_input_receiver_fn to pass to exporters
            * build the estimator and its specs

        """
        super().build()
        project.create_or_get_active_project(
            project_type="train",
            project_dir=self.project_dir,
            continue_last=self.run_config.continue_training)
        self.project_dirs = project.get_active_project_dirs()
        self._model_handler = ModelHandler(
            model=self.model,
            devices=self.run_config.devices,
            global_optimization_config=self.optimization_config,
            variable_strategy=self.run_config.variable_strategy,
            predictions_have_variable_shape=
            self.run_config.predictions_have_variable_shape,
            max_outputs_tb=self.save_config.max_outputs_tb).build()
        dataset_fn_inference = partial(
            self._get_batch_data, mode=tf.estimator.ModeKeys.EVAL)
        self._server_input_receiver_fn_builder = (
            exporters.ServingInputReceiverFnBuilder(
                model=self.model,
                save_dir_inference_graph=self.project_dirs.checkpoints,
                save_dir_inputs_outputs_mapping=self.project_dirs.checkpoints,
                dataset_fn=dataset_fn_inference,
                inference_inputs_have_variable_shape=
                self.save_config.inference_inputs_have_variable_shape))
        self._build_model_dna()
        self._build_callbacks_handler_dna()
        self._set_callback_properties()
        self._set_dataset_modes()
        self._build_estimator()
        self._build_estimator_specs()
        return self

    @object_utils.assert_is_built
    @mlflow_utils.create_mlflow_experiment_and_start_run
    @project_serializer.serialize_run_configs(
        "train_run",
        single_config_names=["trainer", "model"],
        serializers_to_use=[project_serializer.RunConfigSerializer,
                            project_serializer.MlflowConfigSerializer])
    @project_artifacts.serialize_project_artifacts
    @mlflow_utils.log_project_artifacts_to_mlflow
    def run(self):
        """
        Start training and evaluation loop according to provided configuration.
        Will execute :obj:`tf.estimator.train_and_evaluate` using built
        estimator and training and evaluation specs
        """
        tf.estimator.train_and_evaluate(self._estimator,
                                        self._estimator_train_spec,
                                        self._estimator_eval_spec)
        self.clear_caches()

    def clear_caches(self):
        """
        Clear dataset cache files, if dataset has some
        """
        for each_dataset in self.datasets.values():
            each_dataset.clear_cache()

    def _get_dataset_nucleotide(self) -> Dataset:
        train_dataset = self.datasets.get('train')
        eval_dataset = self.datasets.get('eval')
        dataset_nucleotide = train_dataset or eval_dataset
        return dataset_nucleotide

    def _set_callback_properties(self):
        for mode in [tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL]:
            number_iterations_per_epoch = (
                self.run_config.iterations_per_epoch[mode])
            callback_log_dir = os.path.join(self.project_dirs.callbacks, mode)
            self.set_callback_properties(mode, callback_log_dir,
                                         number_iterations_per_epoch)

    def _set_dataset_modes(self):
        for each_mode in self.datasets:
            self.datasets[each_mode].mode = each_mode

    def _build_model_dna(self):
        dataset_nucleotide = self._get_dataset_nucleotide()
        self.model.build_dna(incoming_nucleotides=dataset_nucleotide)

    def _build_callbacks_handler_dna(self):
        dataset_nucleotide = self._get_dataset_nucleotide()
        model_and_dataset_nucleotides = list(
            self.model.all_nucleotides.values())
        model_and_dataset_nucleotides.append(dataset_nucleotide)
        for mode in [tf.estimator.ModeKeys.TRAIN,
                     tf.estimator.ModeKeys.EVAL]:
            self.callbacks_handler[mode].build_dna(
                incoming_nucleotides=model_and_dataset_nucleotides)

    def _build_estimator(self):
        iters_per_epoch = self.run_config.iterations_per_epoch
        model_fn = self._model_handler.get_model_fn()
        run_config = tf.estimator.RunConfig(
            tf_random_seed=self.run_config.random_seed,
            save_checkpoints_steps=iters_per_epoch['train'],
            save_checkpoints_secs=None,
            save_summary_steps=0,
            log_step_count_steps=None,
            session_config=tf.ConfigProto(**self.session_config))

        self._estimator = tf.estimator.Estimator(
            model_fn=model_fn, model_dir=self.project_dirs.checkpoints,
            config=run_config)

    def _build_estimator_specs(self):
        iters_per_epoch = self.run_config.iterations_per_epoch
        max_steps_train = (iters_per_epoch['train'] *
                           self.run_config.num_epochs)
        input_fn_train = partial(self._get_batch_data,
                                 mode=tf.estimator.ModeKeys.TRAIN)
        input_fn_eval = partial(self._get_batch_data,
                                mode=tf.estimator.ModeKeys.EVAL)

        train_hooks = self._get_hooks(tf.estimator.ModeKeys.TRAIN)
        eval_hooks = self._get_hooks(tf.estimator.ModeKeys.EVAL)
        estimator_exporters = self._get_exporters(
            self.callbacks_handler[tf.estimator.ModeKeys.EVAL])

        self._estimator_train_spec = tf.estimator.TrainSpec(
            input_fn_train, max_steps=max_steps_train, hooks=train_hooks)
        self._estimator_eval_spec = tf.estimator.EvalSpec(
            input_fn_eval, steps=iters_per_epoch['eval'],
            hooks=eval_hooks, start_delay_secs=0, throttle_secs=10,
            exporters=estimator_exporters)

    def _get_batch_data(self, mode: str) -> tf.data.Dataset:
        """
        Build the inputs from datasets

        Parameters
        ----------
        mode
            mode of dataset

        Returns
        -------
        data
            dataset with batch of data
        """
        if (mode == tf.estimator.ModeKeys.EVAL and run_utils.is_local_run() and
                self.datasets[mode].cache_dir is not None):
            # remove caching for local run and evaluation
            logger = logging.getLogger(__name__)
            logger.warning('Cache will be disabled for evaluation dataset '
                           'as it is running on local machine!')
            self.datasets[mode].cache_dir = None

        with tf.variable_scope(ScopeNames.DATASET), tf.device('/cpu:0'):
            data = self.datasets[mode](self.run_config.batch_size[mode])
        return data

    def _get_exporters(
            self, callback_handler: CallbacksHandler = None
    ) -> List[exporters.ModelExporter]:
        """
        Get exporters for model

        Parameters
        ----------
        callback_handler
            callback handler for evaluation; if they have kpi
            inside (they were converted to callbacks from :obj:`KPIEvaluator`),
            then kpi value will be added to exporter

        Returns
        -------
        exporters
            list of exporters; currently only 1 exporter is used
        """
        serving_input_receiver_fn = self._server_input_receiver_fn_builder.get()
        if callback_handler is not None:
            kpi_evaluators = callback_handler.kpi_evaluators
        else:
            kpi_evaluators = None
        exporter = exporters.ModelExporter(
            'Exporter', serving_input_receiver_fn=serving_input_receiver_fn,
            exports_dir=self.project_dirs.saved_models,
            kpi_evaluators=kpi_evaluators,
            exports_to_keep=self.save_config.exports_to_keep)
        return [exporter]

    def _get_hooks(self, mode: str) -> List[tf.train.SessionRunHook]:
        """
        Create hooks:

            - checkpoint saver for evaluation
            - metrics update hook for evaluation
            - from callbacks_handler

        """
        iterations_per_epoch = self.run_config.iterations_per_epoch
        hooks = list()
        initializer_hook = self._get_initializer_hook(mode)
        early_stopping_init_hook = session_run_hooks.EarlyStoppingInitHook()
        hooks.extend([initializer_hook, early_stopping_init_hook])
        if mode == tf.estimator.ModeKeys.EVAL:
            summary_hook = self._get_summary_hook(mode)
            metrics_hook = session_run_hooks.MetricUpdateHook()
            hooks.extend([metrics_hook, summary_hook])

        if self.run_config.profile_hook_config:
            profile_hook = tf.train.ProfilerHook(
                output_dir=self.project_dir,
                **self.run_config.profile_hook_config)
            hooks.append(profile_hook)

        max_number_of_iterations_per_epoch = max(iterations_per_epoch.values())
        hooks.append(
            session_run_hooks.convert_callbacks_handler_to_session_hook(
                callbacks_handler=self.callbacks_handler[mode],
                summary_dir=self.project_dirs.summaries,
                max_number_of_iterations_per_epoch=
                max_number_of_iterations_per_epoch
            ))

        if mode == tf.estimator.ModeKeys.TRAIN and run_utils.is_chief():
            hooks.extend(self._get_chief_only_hooks())
        return hooks

    def _get_chief_only_hooks(self) -> List[tf.train.SessionRunHook]:
        """
        Create chief only hooks:

            - checkpoint saver
            - plugins restorer
            - metrics updater
        """
        mode = tf.estimator.ModeKeys.TRAIN
        summary_hook = self._get_summary_hook(mode)
        metrics_hook = session_run_hooks.MetricUpdateHook()
        hooks = [metrics_hook, summary_hook]
        return hooks

    def _get_summary_hook(
            self, mode: str) -> session_run_hooks.SummarySaverHook:
        """
        Create summary saver hook
        """
        iter_per_epoch = self.run_config.iterations_per_epoch
        summary_dir = os.path.join(self.project_dirs.summaries, mode)
        save_steps = self.save_config.save_summary_steps[mode]
        max_iter = max(iter_per_epoch.values())
        step_offset = (iter_per_epoch[mode]
                       if iter_per_epoch[mode] != max_iter else 0)

        summary_hook = session_run_hooks.SummarySaverHook(
            save_steps=save_steps, output_dir=summary_dir,
            step_offset=step_offset)
        return summary_hook

    def _get_initializer_hook(
            self, mode: str
    ) -> session_run_hooks.CustomNucleotideInitializerHook:
        """
        Create custom initializer hook
        """
        all_nucleotides_for_mode = self.get_all_nucleotides(mode)
        initializer_hook = session_run_hooks.CustomNucleotideInitializerHook(
            all_nucleotides_for_mode)
        return initializer_hook
