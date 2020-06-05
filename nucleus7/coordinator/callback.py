# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Interface for callbacks inside of coordinator
"""
import abc
# pylint: disable=unused-import
# is used for typing
from typing import Optional

import tensorflow as tf

from nucleus7.coordinator.configs import RunIterationInfo
from nucleus7.core.nucleotide import Nucleotide
from nucleus7.utils import log_utils
from nucleus7.utils import mlflow_utils
from nucleus7.utils import object_utils


class CoordinatorCallback(Nucleotide):
    """Interface for callback used with coordinator

    Attributes
    ----------
    register_name_scope
        name scope for register and for the config logger
    """

    register_name_scope = "callback"
    exclude_from_register = True

    _process_method_name = "on_iteration_end"

    def __init__(self, *,
                 inbound_nodes,
                 name=None,
                 incoming_keys_mapping=None):
        super(CoordinatorCallback, self).__init__(
            inbound_nodes=inbound_nodes, name=name,
            incoming_keys_mapping=incoming_keys_mapping)
        self._iteration_info = RunIterationInfo(0, 0, 0.0)
        self._number_iterations_per_epoch = None  # type: Optional[int]
        self._log_dir = None  # type: Optional[str]
        self._summary_writer = None  # type: Optional[tf.summary.FileWriter]
        self._summary_step = None  # type: Optional[int]

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

    @mlflow_utils.log_nucleotide_exec_time_to_mlflow(method_name="begin")
    def begin(self):
        """
        This method is be called before the main run, e.g. before tensorflow
        session is created
        """

    @mlflow_utils.log_nucleotide_exec_time_to_mlflow(
        method_name="on_iteration_start")
    def on_iteration_start(self):
        """
        This method is called before each iteration
        """

    @abc.abstractmethod
    def on_iteration_end(self, **data):
        """
        Execute this function after each iteration

        Parameters
        ----------
        **data : dict
            keyword arguments for data
        """

    @mlflow_utils.log_nucleotide_exec_time_to_mlflow(method_name="end")
    def end(self):
        """
        This method is called at the end of the run, e.g. when tensorflow
        session is already closed
        """

    @object_utils.assert_is_built
    @mlflow_utils.log_nucleotide_exec_time_to_mlflow(
        method_name="on_iteration_end")
    @log_utils.log_nucleotide_inputs_outputs()
    @object_utils.raise_exception_with_class_name
    def __call__(self, **inputs):
        # pylint: disable=arguments-differ
        # parent __call__ method has more generic signature
        return self.on_iteration_end(**inputs)
