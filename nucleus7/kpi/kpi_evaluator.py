# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Interface for kpi evaluator
"""
import itertools
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import numpy as np

from nucleus7.core.gene_handler import GeneHandler
from nucleus7.kpi.accumulator import KPIAccumulator
from nucleus7.kpi.kpi_plugin import KPIPlugin
from nucleus7.kpi.saver import KPISaver
from nucleus7.utils import mlflow_utils
from nucleus7.utils import object_utils


class KPIEvaluator(GeneHandler):
    """
    Class for evaluation of KPI metrics.

    Parameters
    ----------
    plugins
        kpi plugins
    accumulators
        kpi accumulators
    """

    nucleotide_type_dependency_map = {
        KPIPlugin: [
            KPIPlugin
        ],
        KPIAccumulator: [
            KPIPlugin,
            KPIAccumulator
        ],
    }
    gene_name_and_nucleotide_super_cls = {
        "plugins": KPIPlugin,
        "accumulators": KPIAccumulator,
    }

    def __init__(self, plugins: Optional[Union[List[KPIPlugin],
                                               Dict[str, KPIPlugin],
                                               KPIPlugin]] = None,
                 accumulators: Optional[Union[List[KPIAccumulator],
                                              Dict[str, KPIAccumulator],
                                              KPIAccumulator]] = None):
        self.plugins = None  # type: Dict[str, KPIPlugin]
        self.accumulators = None  # type: Dict[str, KPIAccumulator]
        super().__init__(plugins=plugins, accumulators=accumulators)
        self._save_target = None
        self._cache_target = None
        self._is_last_sample = False
        self._is_last_iteration = False
        self._last_kpi = None

    @property
    def name(self):
        """
        Name of the kpi evaluator
        """
        return "kpi_evaluator"

    @property
    def last_kpi(self) -> dict:
        """
        Returns
        -------
        last_evaluated_kpi
            last evaluated kpi
        """
        return self._last_kpi or {}

    @property
    def is_last_sample(self) -> bool:
        """
        Returns
        -------
        is_last_sample
            if the sample to proceed is the last one
        """
        return self._is_last_sample

    @is_last_sample.setter
    def is_last_sample(self, is_last_sample: bool):
        self._is_last_sample = is_last_sample
        self._set_property_for_all_genes("is_last_sample", is_last_sample)

    @property
    def is_last_iteration(self) -> bool:
        """
        Returns
        -------
        is_last_iteration
            flag if it is last iteration
        """
        return self._is_last_iteration

    @is_last_iteration.setter
    def is_last_iteration(self, is_last_iteration: bool):
        self._is_last_iteration = is_last_iteration
        self._set_property_for_all_genes("is_last_iteration", is_last_iteration)

    @property
    def save_target(self):
        """
        Save target like directory or server address etc.
        """
        return self._save_target

    @save_target.setter
    def save_target(self, save_target):
        self._save_target = save_target
        self._set_property_for_all_genes("save_target", save_target)

    @property
    def cache_target(self):
        """
        Cache target like cache directory or server address etc.
        """
        return self._cache_target

    @cache_target.setter
    def cache_target(self, cache_target):
        self._cache_target = cache_target
        self._set_property_for_all_genes("cache_target", cache_target)

    def add_saver(self, saver: KPISaver, kpi_gene_name="accumulators"):
        """
        Add saver to all kpi nucleotides from kpi_gene_name

        Parameters
        ----------
        saver
            saver to add
        kpi_gene_name
            gene name to add to, like accumulators or plugins

        Raises
        ------
        AssertionError
            if kpi_gene_name not in [plugins, accumulators]
        """
        assert kpi_gene_name in ["plugins", "accumulators"], (
            "Gene name must be one of [plugins, accumulators]")
        for each_item in getattr(self, kpi_gene_name).values():
            each_item.savers.append(saver)

    @object_utils.assert_is_built
    @mlflow_utils.log_nucleotide_exec_time_to_mlflow(
        method_name="kpi_plugins")
    def process_plugins(self, **inputs) -> dict:
        """
        Process all kpi plugins

        Parameters
        ----------
        inputs
            inputs to kpi plugins

        Returns
        -------
        plugins_results
            results from kpi plugins
        """
        return self.process_gene(gene_name='plugins',
                                 gene_inputs=inputs)

    @object_utils.assert_is_built
    @mlflow_utils.log_nucleotide_exec_time_to_mlflow(
        method_name="kpi_accumulators")
    def process_accumulators(self, **inputs) -> dict:
        """
        Process all kpi accumulators

        Parameters
        ----------
        inputs
            inputs to kpi accumulators

        Returns
        -------
        accumulators_results
            results from kpi accumulators
        """
        return self.process_gene(gene_name='accumulators',
                                 gene_inputs=inputs)

    def clear_state(self):
        """
        Clear states of all nucleotides
        """
        for each_item in itertools.chain(self.plugins.values(),
                                         self.accumulators.values()):
            each_item.clear_state()
        self._last_kpi = None

    @object_utils.assert_is_built
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
        plugin_outputs = self.process_plugins(**inputs)
        kpi = self.process_accumulators(**inputs, **plugin_outputs)
        self._get_last_accumulated_kpi()
        return kpi

    def _get_last_accumulated_kpi(self):
        last_kpi = {}
        for each_accumulator in self.accumulators.values():
            each_accumulator.finalize()
            each_last_kpi = each_accumulator.last_kpi
            if each_last_kpi is not None:
                last_kpi[each_accumulator.name] = each_last_kpi
        self._last_kpi = last_kpi
