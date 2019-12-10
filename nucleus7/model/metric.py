# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Interface for model metric
"""

import abc
from typing import Dict
from typing import List
from typing import Optional

import tensorflow as tf

from nucleus7.model.summary import ModelSummary
from nucleus7.utils import model_utils
from nucleus7.utils import object_utils


class ModelMetric(ModelSummary):
    """
    Model Metric interface

    Will write summaries for all results, which start with prefixes like
    `image_`, `scalar_`, ...

    Parameters
    ----------
    keys_to_exclude_from_sample_mask
        list of keys that will not be masked using sample_mask

    Attributes
    ----------
    register_name_scope
        name scope for register and for the config logger
    """
    register_name_scope = "model_metric"
    exclude_from_register = True

    def __init__(self, *,
                 keys_to_exclude_from_sample_mask: Optional[List[str]] = None,
                 **nucleotide_kwargs):
        super(ModelMetric, self).__init__(**nucleotide_kwargs)
        self.keys_to_exclude_from_sample_mask = keys_to_exclude_from_sample_mask

    # pylint: disable=no-self-argument
    # classproperty is class property and so cls must be used
    @object_utils.classproperty
    def incoming_keys_optional(cls):
        extra_keys = ['sample_mask']
        return super().incoming_keys_optional + extra_keys

    @abc.abstractmethod
    def process(self, **inputs) -> Dict[str, tf.Tensor]:
        """
        Calculate metrics given inputs

        Parameters
        ----------
        **inputs
            dict with all tensors (and ground truth tensors) as values

        Returns
        ------
        metrics
            metrics
        """

    def _call(self, sample_mask=None, **inputs) -> Dict[str, tf.Tensor]:
        """
        wrapper on top of `self.process` with variable scope
        """
        # pylint: disable=arguments-differ
        # parent _call method has more generic signature
        if sample_mask is not None:
            inputs = model_utils.select_inputs_by_sample_mask(
                sample_mask, self.keys_to_exclude_from_sample_mask, **inputs)
        metric = self.process(**inputs)
        self.warn_about_not_stored_summaries(metric)
        return metric
