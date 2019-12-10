# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Interface for Callback handler
"""
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import numpy as np
import tensorflow as tf

from nucleus7.coordinator.callback import CoordinatorCallback
from nucleus7.coordinator.configs import RunIterationInfo
from nucleus7.core.dna_helix import DNAHelix
from nucleus7.core.gene_handler import GeneHandler
# pylint: disable=cyclic-import
from nucleus7.kpi.kpi_callback import KPIEvaluatorCallback
# pylint: enable=cyclic-import
from nucleus7.kpi.kpi_evaluator import KPIEvaluator
from nucleus7.utils import mlflow_utils
from nucleus7.utils import object_utils


class CallbacksHandler(GeneHandler):
    """
    Interface to hold the collection of callbacks

    Parameters
    ----------
    callbacks
        collection of callbacks
    """
    nucleotide_type_dependency_map = {
        CoordinatorCallback: [CoordinatorCallback]
    }
    gene_name_and_nucleotide_super_cls = {
        'callbacks': CoordinatorCallback
    }

    def __init__(self, callbacks: Union[List[CoordinatorCallback],
                                        Dict[str, CoordinatorCallback],
                                        CoordinatorCallback]):
        self.callbacks = None  # type: Dict[str, CoordinatorCallback]
        super().__init__(callbacks=callbacks)
        self._iteration_info = RunIterationInfo(0, 0, 0.0)
        self._number_iterations_per_epoch = None  # type: Optional[int]
        self._log_dir = None  # type: Optional[str]
        self._summary_writer = None  # type: Optional[tf.summary.FileWriter]
        self._summary_step = None  # type: Optional[int]

    @property
    def name(self):
        """
        Name of the callback handler

        Returns
        -------
        name
            name of the handler
        """
        return 'callback_handler'

    @property
    def iteration_info(self) -> RunIterationInfo:
        """
        Iteration info

        Returns
        -------
        iteration_info
            iteration info
        """
        return self._iteration_info

    @iteration_info.setter
    def iteration_info(self, iteration_info: RunIterationInfo):
        self._iteration_info = iteration_info
        for callback in self.callbacks.values():
            callback.iteration_info = self._iteration_info

    @property
    def number_iterations_per_epoch(self) -> int:
        """
        Number of iterations per epoch

        Returns
        -------
        number_iterations_per_epoch
            number of iterations per epoch
        """
        return self._number_iterations_per_epoch

    @number_iterations_per_epoch.setter
    def number_iterations_per_epoch(self, number_iterations_per_epoch: int):
        self._number_iterations_per_epoch = number_iterations_per_epoch
        self._set_property_for_all_genes(
            'number_iterations_per_epoch', number_iterations_per_epoch)

    @property
    def log_dir(self) -> str:
        """
        Log directory for callback

        Returns
        -------
        log_dir
            path to log directory
        """
        return self._log_dir

    @log_dir.setter
    def log_dir(self, log_dir: str):
        self._log_dir = log_dir
        self._set_property_for_all_genes('log_dir', log_dir)

    @property
    def summary_writer(self) -> tf.summary.FileWriter:
        """
        Summary writer

        Returns
        -------
        summary_writer
            summary writer
        """
        return self._summary_writer

    @summary_writer.setter
    def summary_writer(self, summary_writer: tf.summary.FileWriter):
        self._summary_writer = summary_writer
        self._set_property_for_all_genes('summary_writer', summary_writer)

    @property
    def summary_step(self) -> int:
        """
        Current summary step

        Returns
        -------
        summary_step
            summary step
        """
        return self._summary_step

    @summary_step.setter
    def summary_step(self, summary_step: int):
        self._summary_step = summary_step
        self._set_property_for_all_genes('summary_step', summary_step)

    @property
    def kpi_evaluators(self) -> List[KPIEvaluator]:
        """
        KPI Evaluator callbacks

        Returns
        -------
        evaluator_callbacks
            list of callbacks with kpi evaluator inside
        """
        kpi_evaluators = []
        for each_callback in self.callbacks.values():
            if isinstance(each_callback, KPIEvaluatorCallback):
                kpi_evaluators.append(each_callback.evaluator)
        return kpi_evaluators

    @property
    def kpi_evaluators_dna_helices(self) -> Optional[Dict[str, DNAHelix]]:
        """
        Get the dna helices for kpi evaluators if they exist

        Returns
        -------
        kpi_evaluators_dna_helices
            dict with mapping kpi evaluator name to its dna helix
        """
        kpi_evaluators = self.kpi_evaluators
        if not kpi_evaluators:
            return None

        return {each_evaluator.name: each_evaluator.dna_helix
                for each_evaluator in kpi_evaluators}

    def filter_inputs(self, inputs: dict) -> dict:
        """
        Filter the inputs by inbound nodes

        Parameters
        ----------
        inputs
            all inputs

        Returns
        -------
        inputs_filtered_by_inbound_nodes
            inputs with only keys from inbound_nodes
        """
        inputs_filtered_by_inbound_nodes = {
            each_key: inputs[each_key]
            for each_key in self.inbound_nodes
        }
        return inputs_filtered_by_inbound_nodes

    @object_utils.assert_is_built
    @mlflow_utils.log_nucleotide_exec_time_to_mlflow(
        method_name="begin")
    def begin(self):
        """
        This method is be called before the tensorflow session is generated
        """
        for _, each_callback in sorted(self.callbacks.items()):
            each_callback.begin()

    @object_utils.assert_is_built
    @mlflow_utils.log_nucleotide_exec_time_to_mlflow(
        method_name="on_iteration_start")
    def on_iteration_start(self):
        """
        This method is called before each iteration
        """
        for _, each_callback in sorted(self.callbacks.items()):
            each_callback.on_iteration_start()

    @object_utils.assert_is_built
    @mlflow_utils.log_nucleotide_exec_time_to_mlflow(
        method_name="on_iteration_end")
    def __call__(self, **inputs: Dict[str, np.ndarray]) -> dict:
        """
        Call the callbacks on inputs

        Parameters
        ----------
        inputs
            dict with inputs

        Returns
        -------
        callback_results
            callback results
        """
        # pylint: disable=arguments-differ
        # parent __call__ method has more generic signature
        return self.process_gene(gene_name='callbacks',
                                 gene_inputs=inputs)

    @object_utils.assert_is_built
    @mlflow_utils.log_nucleotide_exec_time_to_mlflow(method_name="end")
    def end(self):
        """
        This method is called when tensorflow session is closed
        """
        for _, each_callback in sorted(self.callbacks.items()):
            each_callback.end()
