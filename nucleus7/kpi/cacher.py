# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Interface to use to cache the KPI values
"""
import abc
import hashlib
import json
import logging
import os
from typing import Optional

import numpy as np

from nucleus7.core.base import BaseClass
from nucleus7.utils import io_utils
from nucleus7.utils import nest_utils
from nucleus7.utils import object_utils


class KPICacher(BaseClass):
    """
    Interface to use to cache KPI values
    """

    def __init__(self):
        super(KPICacher, self).__init__()
        self._current_hash = None
        self._cache_target = None

    @property
    def cache_target(self):
        """
        Cache target like cache directory or server address etc.
        """
        return self._cache_target

    @cache_target.setter
    def cache_target(self, cache_target):
        if self._cache_target is None or cache_target is None:
            self._cache_target = cache_target
        else:
            raise ValueError("Cannot set cache_target since it is already set! "
                             "Set it to None before!")

    @property
    def current_hash(self):
        """
        last calculated cache
        """
        return self._current_hash

    @abc.abstractmethod
    def calculate_hash(self, inputs, cache_prefix: Optional[str] = None):
        """
        Calculate hash value for inputs and cache prefix and assign
        the hash value to self._current_hash

        Parameters
        ----------
        inputs
            inputs to for kpi calculation
        cache_prefix
            additional prefix to use for hash value calculation
        """

    @abc.abstractmethod
    def cache(self, values):
        """
        Cache values

        Parameters
        ----------
        values
            values to cache
        """

    @abc.abstractmethod
    def restore(self):
        """
        Restore values from cache using self.current_hash

        Returns
        -------
        kpi_restored
            restored value of KPIs from cache
        """


class KPIMD5Cacher(KPICacher):
    """
    KPI Cacher uses md5 hash on the json representation of inputs together with
    cache prefix. If some value inside of inputs cannot be represented inside
    of json, then method `value.tostring()` is used if it exists, otherwise
    simple `str(value)` is used. To change only this behavior, override
    `_default_decoder_fn` method.
    """

    def __init__(self, name: str = "md5_cacher"):
        super().__init__()
        self.name = name

    def calculate_hash(self, inputs, cache_prefix: Optional[str] = None):
        inputs_flatten = nest_utils.flatten_nested_struct(inputs)
        json_str = json.dumps(inputs_flatten, indent=2, sort_keys=True,
                              default=self._default_decoder_fn)
        md5_hash = hashlib.md5()
        md5_hash.update(str(cache_prefix).encode())
        md5_hash.update(json_str.encode())
        if cache_prefix:
            self._current_hash = "-".join([str(cache_prefix),
                                           md5_hash.hexdigest()])
        else:
            self._current_hash = md5_hash.hexdigest()

    @object_utils.assert_property_is_defined("current_hash")
    @object_utils.assert_is_built
    def cache(self, values):
        io_utils.maybe_mkdir(self.cache_target)
        cache_fname = self._get_cache_fname()
        inputs_flatten = nest_utils.flatten_nested_struct(values)
        if os.path.exists(cache_fname):
            logger = logging.getLogger(__name__)
            logger.warning("Cache with name %s already exist!", cache_fname)
            return

        with open(cache_fname, "w") as file:
            json.dump(inputs_flatten, file, default=lambda x: x.tolist())

    @object_utils.assert_property_is_defined("current_hash")
    @object_utils.assert_is_built
    def restore(self):
        io_utils.maybe_mkdir(self.cache_target)
        cache_fname = self._get_cache_fname()
        if not os.path.exists(cache_fname):
            return None
        with open(cache_fname, "r") as file:
            restored = json.load(file)
        for each_key in restored:
            if isinstance(restored[each_key], list):
                restored[each_key] = np.array(restored[each_key])
        restored_unflatten = nest_utils.unflatten_dict_to_nested(restored)
        logger = logging.getLogger(__name__)
        logger.debug("restoring KPI values from %s", cache_fname)
        return restored_unflatten

    def _get_cache_fname(self):
        cache_fname = "_".join([self.name, self._current_hash]) + ".json"
        full_cache_fname = os.path.join(self.cache_target, cache_fname)
        return full_cache_fname

    def _default_decoder_fn(self, value):
        # pylint: disable=no-self-use
        # is an interface
        try:
            return str(value.tostring())
        except AttributeError as e:  # pylint: disable=invalid-name
            raise AttributeError(
                "Only json serializable objects values or np.ndarray are "
                "allowed to be used with KPIMD5Cacher! ({})".format(e))
