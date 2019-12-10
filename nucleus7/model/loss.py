# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Interface for model loss
"""

import abc
import logging
from typing import Dict
from typing import List
from typing import Optional
import warnings

import tensorflow as tf

from nucleus7.core.nucleotide import TfNucleotide
from nucleus7.utils import deprecated
from nucleus7.utils import model_utils
from nucleus7.utils import object_utils


class ModelLoss(TfNucleotide):
    """
    Model loss interface

    Parameters
    ----------
    loss_weights
        defines the coefficients of loss terms in total_loss
        `total_loss += sum({loss_weights[k]*l for k, l in loss.items()})`
    exclude_from_total_loss
        if this loss should be excluded from total loss; can be used to scale or
        use it further in other losses
    keys_to_exclude_from_sample_mask
        list of keys that will not be masked using sample_mask

    Attributes
    ----------
    register_name_scope
        name scope for register and for the config logger
    """
    register_name_scope = "model_loss"
    exclude_from_register = True

    @deprecated.replace_deprecated_parameter('loss_scale_factors',
                                             'loss_weights',
                                             required=False)
    def __init__(self, *,
                 loss_weights: Optional[dict] = None,
                 exclude_from_total_loss: bool = False,
                 keys_to_exclude_from_sample_mask: Optional[List[str]] = None,
                 **nucleotide_kwargs):
        super(ModelLoss, self).__init__(**nucleotide_kwargs)
        self.loss_weights = loss_weights or {}
        self.exclude_from_total_loss = exclude_from_total_loss
        self.keys_to_exclude_from_sample_mask = keys_to_exclude_from_sample_mask

    @abc.abstractmethod
    def process(self, **inputs) -> Dict[str, tf.Tensor]:
        """
        Calculate the loss given inputs

        Parameters
        ----------
        inputs
            dict with input tensors

        Returns
        ------
        loss
            loss values
        """

    # pylint: disable=no-self-argument
    # classproperty is class property and so cls must be used
    @object_utils.classproperty
    def incoming_keys_optional(cls):
        extra_keys = ['sample_mask']
        return super().incoming_keys_optional + extra_keys

    # pylint: disable=no-self-argument
    # classproperty is class property and so cls must be used
    @object_utils.classproperty
    def generated_keys_optional(cls):
        extra_keys = ['total_loss']
        return super().generated_keys_optional + extra_keys

    def _call(self, *,
              sample_mask: tf.Tensor = None,
              **inputs) -> Dict[str, tf.Tensor]:
        """
        Call the loss on inputs. Will add masking, combine losses to total_loss
        and multiply the single losses by corresponding loss scale factor

        Parameters
        ----------
        sample_mask
            mask to apply on loss; if specified, the loss is multiplied with
            it, e.g. for multiple heads
        **inputs
            all inputs to loss

        Returns
        ------
        losses
            dict with losses
        """
        # pylint: disable=arguments-differ
        # parent __call__ method has more generic signature
        logger = logging.getLogger(__name__)
        if sample_mask is not None:
            if self.generated_keys_optional != ["total_loss"]:
                warnings.warn(
                    "You use loss masking and also have optional generated "
                    "keys inside of {}. In case of no samples are masked, "
                    "it may cause errors since loss will return 0 for all "
                    "generated keys".format(self.name))
            losses = self._process_masked(sample_mask, **inputs)
        else:
            losses = self.process(**inputs)
        total_loss = 0.0
        for loss_name, loss in losses.items():
            if loss_name.startswith('loss'):
                scale_factor = self.loss_weights.get(loss_name, 1.0)
                total_loss += scale_factor * loss

        if not self.exclude_from_total_loss:
            losses['total_loss'] = total_loss
        else:
            logger.warning("Loss %s will be excluded from total loss",
                           self.name)
        return losses

    def _process_masked(self, sample_mask, **inputs):

        def _process_if_not_empty():
            inputs_masked = model_utils.select_inputs_by_sample_mask(
                sample_mask, self.keys_to_exclude_from_sample_mask, **inputs)
            losses_ = self.process(**inputs_masked)
            return losses_

        def _get_zero_losses():
            keys = [each_key for each_key in self.generated_keys_all
                    if each_key != "total_loss"]

            return {each_key: tf.constant(0.0) for each_key in keys}

        num_masked_samples = tf.reduce_sum(tf.cast(sample_mask, tf.int32))
        losses = tf.cond(tf.greater(num_masked_samples, 0),
                         _process_if_not_empty,
                         _get_zero_losses)
        return losses
