# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Configs for model
"""

from collections import namedtuple

ModelResults = namedtuple('ModelResults',
                          ['inputs_preprocessed', 'predictions_raw',
                           'predictions', 'losses', 'summary', 'metrics',
                           'grads_and_vars', 'regularization_grads_and_vars'])


class MixedPrecisionConfig(namedtuple('MixedPrecisionConfig',
                                      ['use', 'loss_scale_factor'])):
    """
    Configuration for mixed precision during training

    Parameters
    ----------
    use : boolean, default: False
        if the mixed precision training should be used; in that case, all
        tensors are casted and stored to float16 and variables are stored in
        float32, but are casted to float16 for forward pass; also losses
        are scaled with mixed_precision_loss_scale_factor and then gradients
        are downscaled with same factor; exported model will have
        original precision
    loss_scale_factor : float, default: 128.0
        factor to scale loss and gradients in case of mixed precision training
    """

    def __new__(cls, use: bool = False, loss_scale_factor: float = 128.0):
        return super(MixedPrecisionConfig, cls).__new__(
            cls, use, loss_scale_factor)
