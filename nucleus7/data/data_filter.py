# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
DataFilter interface to use with data objects, like FileList and Dataset
"""
import abc
from typing import List
from typing import Optional

import tensorflow as tf

from nucleus7.core.base import BaseClass
from nucleus7.core.base import MetaLogAndRegister
from nucleus7.utils import nucleotide_utils


class DataFilter(BaseClass, metaclass=MetaLogAndRegister):
    """
    Interface to use to filter the data

    Parameters
    ----------
    predicate_keys_mapping
        mapping of input data keys to predicate input keys
    """
    register_name_scope = "data_filter"
    exclude_from_register = True

    def __init__(self,
                 predicate_keys_mapping: Optional[dict] = None):
        super().__init__()
        self.predicate_keys_mapping = predicate_keys_mapping or {}

    def remap_inputs(self, **inputs) -> dict:
        """
        Remap inputs according to self.predicate_keys_mapping

        Parameters
        ----------
        inputs
            inputs to remap

        Returns
        -------
        inputs_remapped
            inputs remapped using self.predicate_keys_mapping
        """
        inputs_remapped = nucleotide_utils.remap_and_collapse_inputs(
            [inputs], [self.predicate_keys_mapping])
        return inputs_remapped

    @abc.abstractmethod
    def predicate(self, **inputs) -> bool:
        """
        Predicate function. It will leave the samples where predicate() == True
        and filter out all other where predicate() == False.

        Parameters
        ----------
        **inputs
            inputs to filter

        Returns
        -------
        leave_or_filter_out
            bool indicating if the result must be filtered out (if False) or
            used (if True)
        """


class DataFilterMixin:
    """
    Mixin to enable data filter functionality

    Attributes
    ----------
    _data_filters
        will be created by first add of filter and hold data filters
    """

    @property
    def data_filters(self) -> Optional[List[DataFilter]]:
        """
        Returns
        -------
        data_filters
            if self._data_filters exist, returns it, otherwise returns None
        """
        return getattr(self, "_data_filters", None)

    def add_data_filter(self, data_filter: DataFilter):
        """
        Add data filter to object

        Parameters
        ----------
        data_filter
            data filter to add to the object

        Raises
        ------
        ValueError
            if data_filter is not an instance of DataFilter class
        """
        data_filters = self.data_filters
        if not isinstance(data_filter, DataFilter):
            raise ValueError("Is not a DataFilter! ({})".format(data_filter))

        if data_filters is None:
            data_filters = []
        data_filters.append(data_filter)
        self._data_filters = data_filters

    def add_data_filters(self, data_filters: List[DataFilter]):
        """
        Add list of data filters

        Parameters
        ----------
        data_filters
            list of data filters
        """
        for each_data_filter in data_filters:
            self.add_data_filter(each_data_filter)

    def data_filter_true(self, **inputs) -> bool:
        """
        Cumulative predicate function of all the data_filters

        Parameters
        ----------
        **inputs
            inputs to data filter

        Returns
        -------
        filter_flag
            if no data_filters were added, returns all the time True, otherwise
            returns True if all the filters evaluate to True and False otherwise
        """
        if not self.data_filters:
            return True

        all_filter_flags = []
        for each_filter in self.data_filters:
            filter_inputs_remapped = each_filter.remap_inputs(**inputs)
            filter_flag = each_filter.predicate(**filter_inputs_remapped)
            all_filter_flags.append(filter_flag)

        if isinstance(all_filter_flags[0], tf.Tensor):
            return tf.reduce_all(all_filter_flags)

        return all(all_filter_flags)
