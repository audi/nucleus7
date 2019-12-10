# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Classes and methods to convert KPIEvaluator to a CoordinatorCallback
"""

import logging
import os
from typing import Dict
from typing import Optional

import tensorflow as tf

from nucleus7.coordinator import CoordinatorCallback
from nucleus7.core.nucleotide import Nucleotide
from nucleus7.kpi.kpi_evaluator import KPIEvaluator
from nucleus7.kpi.saver import TfSummaryKPISaver
from nucleus7.utils import object_utils


class KPIEvaluatorCallback(CoordinatorCallback):
    """
    Callback with :obj:`KPIEvaluator`

    Parameters
    ----------
    evaluator
        evaluator to use
    num_epochs_to_clean_state
        number of epochs between state of the evaluator will be cleaned
    """
    exclude_from_register = True
    dynamic_incoming_keys = True
    dynamic_generated_keys = True

    def __init__(self, evaluator: KPIEvaluator,
                 inbound_nodes: list,
                 incoming_keys_mapping: dict,
                 num_epochs_to_clean_state: int = 1):

        super().__init__(inbound_nodes=inbound_nodes,
                         incoming_keys_mapping=incoming_keys_mapping,
                         name=evaluator.name)
        self.num_epochs_to_clean_state = num_epochs_to_clean_state
        self.evaluator = evaluator
        self._kpi_summary_saver = None  # type: Optional[TfSummaryKPISaver]

    # pylint: disable=no-self-argument
    # classproperty is class property and so cls must be used
    @object_utils.classproperty
    def incoming_keys_optional(cls):
        extra_keys = ['sample_mask']
        return super().incoming_keys_optional + extra_keys

    @property
    def use_genes_as_inputs(self):
        return True

    @CoordinatorCallback.log_dir.setter  # pylint: disable=no-member
    def log_dir(self, log_dir: str):
        self._log_dir = log_dir
        self.evaluator.save_target = log_dir
        cache_dir = os.path.join(log_dir, "_cache")
        self.evaluator.cache_target = cache_dir

    @CoordinatorCallback.summary_writer.setter  # pylint: disable=no-member
    def summary_writer(self, summary_writer: tf.summary.FileWriter):
        self._summary_writer = summary_writer
        if summary_writer is None:
            return
        self._maybe_create_summary_writer_saver()
        self._kpi_summary_saver.summary_writer = self.summary_writer

    @CoordinatorCallback.summary_step.setter  # pylint: disable=no-member
    def summary_step(self, summary_step: int):
        self._summary_step = summary_step
        self._maybe_create_summary_writer_saver()
        self._kpi_summary_saver.summary_step = self.summary_step

    def build_dna(self, incoming_nucleotides):
        """
        Wrapper to build the dna of evaluator

        Parameters
        ----------
        incoming_nucleotides
            incoming nucleotides to evaluator
        """
        self.evaluator.build_dna(incoming_nucleotides)

    @property
    def all_nucleotides(self) -> Dict[str, Nucleotide]:
        """
        All nucleotides inside of the evaluator

        Returns
        -------
        all_nucleotides
            all nucleotides inside of the evaluator
        """
        return self.evaluator.all_nucleotides

    def on_iteration_end(self, **data):
        # pylint: disable=arguments-differ
        # parent on_iteration_end method has more generic signature
        self._maybe_clear_evaluator_state()
        self.evaluator.is_last_iteration = self.iteration_info.is_last_iteration
        return self.evaluator(**data)

    def _maybe_clear_evaluator_state(self):
        """
        Clear state of estimator if it is a first iteration of epoch when state
        should be cleared
        """
        logger = logging.getLogger(__name__)
        iteration_info = self.iteration_info
        clean_state = (iteration_info.epoch_number > 0
                       and iteration_info.iteration_number <= 1
                       and self.num_epochs_to_clean_state > 0
                       and not (iteration_info.epoch_number
                                % self.num_epochs_to_clean_state))

        if clean_state:
            logger.info('Clear evaluator state according to epoch_number')
            self.evaluator.clear_state()

    def _maybe_create_summary_writer_saver(self):
        if self._kpi_summary_saver is not None:
            return

        self._kpi_summary_saver = TfSummaryKPISaver()
        self.evaluator.add_saver(self._kpi_summary_saver,
                                 kpi_gene_name="accumulators")


def convert_evaluator_to_callback(
        evaluator: KPIEvaluator,
        num_epochs_to_clean_state: int = 1
) -> KPIEvaluatorCallback:
    """
    Convert kpi evaluator to :obj:`CoordinatorCallback`

    If this callback is used inside of training / evaluation, it will add
    TfSummarySaver to savers of the evaluator and will save the kpis to
    main summary writer using KPI accumulators save method. All other savers
    / cachers will be used as usual

    Parameters
    ----------
    evaluator
        evaluator to transform
    num_epochs_to_clean_state
        defines after how many epochs state of evaluator should be cleared

    Returns
    -------
    evaluator_as_callback
        callback transformed from evaluator
    """
    evaluator_as_callback = KPIEvaluatorCallback(
        evaluator=evaluator, inbound_nodes=evaluator.inbound_nodes,
        num_epochs_to_clean_state=num_epochs_to_clean_state,
        incoming_keys_mapping={},
    ).build()
    return evaluator_as_callback
