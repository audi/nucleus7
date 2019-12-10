# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Interface for plugins for KPI evaluation
"""
import abc
import logging
from typing import Callable
from typing import List
from typing import Optional

import numpy as np

from nucleus7.core.nucleotide import Nucleotide
from nucleus7.kpi.cacher import KPICacher
from nucleus7.kpi.saver import KPISaver
from nucleus7.utils import nest_utils
from nucleus7.utils import object_utils
from nucleus7.utils import utils


# pylint: disable=too-many-instance-attributes
# is needed to make the KPIPlugin more generic
class KPIPlugin(Nucleotide):
    """
    Plugin to be used for KPI calculation. Can cache and save results using
    cachers and savers

    Parameters
    ----------
    cachers
        list of cachers or single instance to use to check if KPI values were
        already calculated and if so, use cached ones
    savers
        list of savers or single instance to sve the kpi results once they were
        calculated
    not_batch_keys
        list of keys that will not be masked using sample_mask
    """
    register_name_scope = "kpi_plugin"
    exclude_from_register = True
    exclude_args_from_log = [
        "cachers",
        "savers",
    ]

    def __init__(self,
                 cachers: Optional[List[KPICacher]] = None,
                 savers: Optional[List[KPISaver]] = None,
                 not_batch_keys: Optional[List[str]] = None,
                 **nucleotide_kwargs):
        super().__init__(**nucleotide_kwargs)
        self.cachers = cachers or []
        self.savers = savers or []
        self.not_batch_keys = not_batch_keys or []
        self._save_target = None
        self._cache_target = None
        self._is_last_iteration = False
        self._is_last_sample = False
        self._last_index = 0

    # pylint: disable=no-self-argument
    # classproperty is class property and so cls must be used
    @object_utils.classproperty
    def incoming_keys_optional(cls):
        extra_keys = ["prefix", "sample_mask"]
        return super().incoming_keys_optional + extra_keys

    @property
    def save_target(self):
        """
        Save target is used inside of savers

        Returns
        -------
        save_target
            save_target from plugin
        """
        return self._save_target

    @save_target.setter
    def save_target(self, save_target):
        self._save_target = save_target
        for each_saver in self.savers:
            try:
                each_saver.save_target = save_target
            except ValueError:
                logger = logging.getLogger(__name__)
                logger.info("save_target for saver %s inside of %s is set to "
                            "%s", each_saver, self.name, each_saver.save_target)

    @property
    def cache_target(self):
        """
        Cache target is used inside of cachers

        Returns
        -------
        cache_target
            cache_target from plugin
        """
        return self._cache_target

    @cache_target.setter
    def cache_target(self, cache_target):
        self._cache_target = cache_target
        for each_saver in self.cachers:
            try:
                each_saver.cache_target = cache_target
            except ValueError:
                logger = logging.getLogger(__name__)
                logger.info("cache_target for cacher %s inside of %s is set to "
                            "%s", each_saver, self.name, each_saver.save_target)

    @abc.abstractmethod
    def process(self, **inputs) -> dict:
        """
        Main method for KPIPlugin and must be overridden in particular
        implementations

        Parameters
        ----------
        inputs
            dict with inputs, where each value corresponds not to a batch, but
            to value of singe sample

        Returns
        -------
        kpis
            calculated kpi values
        """

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

    def clear_state(self):
        """
        Clear state of plugin
        """
        self._last_index = 0

    def calculate_hashes(self, inputs, cache_prefix: Optional[str] = None):
        """
        Calculate hash values using all the cachers

        Parameters
        ----------
        inputs
            inputs to calculate the hash
        cache_prefix
            additional prefix for hash
        """
        for each_cacher in self.cachers:
            each_cacher.calculate_hash(inputs=inputs, cache_prefix=cache_prefix)

    def maybe_restore_from_cache(self):
        """
        Restore kpi from cache if it was found by at least one cacher

        Returns
        -------
        restored_kpi
            kpi restored from cacher
        """
        for each_cacher in self.cachers:
            retrieved_from_cache = each_cacher.restore()
            if retrieved_from_cache is not None:
                return retrieved_from_cache
        return None

    def save(self, kpi_values, prefix: Optional[str] = None):
        """
        Save kpi values using all the savers

        Parameters
        ----------
        kpi_values
            kpi values to save
        prefix
            additional prefix for kpi
        """
        for each_saver in self.savers:
            save_name = prefix if each_saver.add_prefix_to_name else self.name
            each_saver.save(name=save_name, values=kpi_values)

    def cache(self, kpi_values):
        """
        Cache kpi values using all cachers and previously calculated hashes

        Parameters
        ----------
        kpi_values
            kpi values to cache
        """
        for each_cacher in self.cachers:
            each_cacher.cache(kpi_values)

    def evaluate_on_batch(self, method_to_evaluate: Callable,
                          sample_mask=None, **inputs):
        """
        Call KPIEvaluator on the batch of data

        Parameters
        ----------
        method_to_evaluate
            method to call on the sample inputs
        sample_mask
            optional batch indicator which samples should be evaluated on
        inputs
            batch inputs to kpi evaluator

        Returns
        -------
        kpi
            calculated kpi
        """
        (batch_inputs_as_list, not_batch_inputs
         ) = utils.split_batch_inputs(inputs,
                                      not_batch_keys=self.not_batch_keys)
        batch_size = len(batch_inputs_as_list)
        is_last_sample_batchwise = utils.get_is_last_sample_batchwise(
            batch_size, self.is_last_iteration, sample_mask)

        list_of_kpis = []
        for i_sample, each_sample_inputs in enumerate(
                batch_inputs_as_list):
            if sample_mask is None or sample_mask[i_sample]:
                self.is_last_sample = is_last_sample_batchwise[i_sample]
                kpi_sample = method_to_evaluate(
                    **each_sample_inputs,
                    **not_batch_inputs)
                if kpi_sample:
                    list_of_kpis.append(kpi_sample)

        if list_of_kpis:
            kpi = nest_utils.combine_nested(list_of_kpis,
                                            combine_fun=np.array)
        else:
            kpi = None
        return kpi

    def evaluate_on_sample(self, *,
                           prefix: Optional[str] = None,
                           **sample_inputs) -> dict:
        """
        Perform evaluation on single sample

        Parameters
        ----------
        prefix
            prefix to use inside of cachers and savers
        sample_inputs
            sample inputs to kpi

        Returns
        -------
        kpi
            evaluated kpi
        """
        full_cache_prefix = self._get_full_prefix(prefix)
        self.calculate_hashes(sample_inputs, cache_prefix=full_cache_prefix)
        kpi_from_cache = self.maybe_restore_from_cache()
        if kpi_from_cache is None:
            kpi = self.process(**sample_inputs)
            self.cache(kpi)
        else:
            kpi = kpi_from_cache
        self.save(kpi, prefix=full_cache_prefix)
        return kpi

    def __call__(self, *, sample_mask=None, **inputs):
        # pylint: disable=arguments-differ
        # parent __call__ method has more generic signature
        return self.evaluate_on_batch(self.evaluate_on_sample,
                                      sample_mask=sample_mask, **inputs)

    def _get_full_prefix(self, cache_prefix: Optional[str] = None):
        if not cache_prefix:
            self._last_index += 1
            cache_prefix = str(self._last_index)

        if isinstance(cache_prefix, str):
            cache_prefix = [cache_prefix]
        full_prefix = "-".join(cache_prefix + [self.name])
        return full_prefix
