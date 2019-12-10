# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Interface for model postprocessor
"""

import abc
from typing import Dict

import tensorflow as tf

from nucleus7.core.nucleotide import TfNucleotide


class ModelPostProcessor(TfNucleotide):
    """
    Model Postprocessor interface

    Attributes
    ----------
    register_name_scope
        name scope for register and for the config logger
    """
    register_name_scope = "model_postprocessor"
    exclude_from_register = True

    @abc.abstractmethod
    def process(self, **inputs) -> Dict[str, tf.Tensor]:
        """
        Apply postprocessing on inputs

        Parameters
        ----------
        **inputs : dict
            dict with input tensors

        Returns
        ------
        predictions_pp : dict
            predictions after applying postprocessing
        """
