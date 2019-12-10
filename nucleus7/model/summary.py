# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Interface for model summary
"""

import abc
import logging
from typing import Dict

import tensorflow as tf

from nucleus7.core.nucleotide import TfNucleotide


class ModelSummary(TfNucleotide):
    """
    Model Summary interface

    Parameters
    ----------
    store_inside_tensorboard
        if the summary should be stored to tensorboard or should only serve
        as inputs to other summaries without storing

    Attributes
    ----------
    register_name_scope
        name scope for register and for the config logger
    _summary_prefixes
        which prefixes are used further to distinguish between summary types
        in tensorboard. Add them to your summaries if you want to have them
        in tensorboard. Otherwise the logging warning will be raised and
        that summaries will not be written
    """
    register_name_scope = "model_summary"
    exclude_from_register = True

    _summary_prefixes = {
        'scalar_',
        'image_',
        'histogram_',
        'text_',
        'audio_',
    }

    def __init__(self, *,
                 store_inside_tensorboard: bool = True,
                 **nucleotide_kwargs):
        super(ModelSummary, self).__init__(**nucleotide_kwargs)
        self.store_inside_tensorboard = store_inside_tensorboard

    @abc.abstractmethod
    def process(self, **inputs) -> Dict[str, tf.Tensor]:
        """
        Get summary given inputs.

        To separate different types of summaries afterwards,
        the prefix is used:

            - `scalar_{}`
            - `image_{}`
            - `histogram_{}`
            - `text_{}`
            - `audio_{}`

        Names without this prefixes will not be included to summaries
        and further to tensorboard.

        Parameters
        ----------
        **inputs
            dict with all tensors (and ground truth tensors) as values

        Returns
        ------
        summaries
            summaries
        """

    def _call(self, **inputs) -> Dict[str, tf.Tensor]:
        """
        wrapper on top of `self.process` with variable scope
        """
        # pylint: disable=arguments-differ
        # parent __call__ method has more generic signature
        summary = self.process(**inputs)
        self.warn_about_not_stored_summaries(summary)
        return summary

    def warn_about_not_stored_summaries(self, summary: dict):
        """
        Warn if there are summaries that will not be stored to tensorboard

        Parameters
        ----------
        summary
            dict with summaries
        """
        logger = logging.getLogger(__name__)
        for summary_name in summary:
            is_summary_ok = False
            for each_prefix in self._summary_prefixes:
                if summary_name.find(each_prefix) >= 0:
                    is_summary_ok = True
                    break
            if not is_summary_ok:
                warn = ("Summary {}/{} will not be used in tensorboard "
                        "as it does not have supported prefix in name!!! "
                        "For further information see :obj:`ModelSummary`"
                        "help".format(self.name, summary_name))
                logger.warning(warn)
