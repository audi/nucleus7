# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Model fields
"""


# pylint: disable=too-few-public-methods
# it is only a container, so no methods required
class ScopeNames:
    """
    Names of the model scopes
    """
    PREPROCESSING = 'preprocessing'
    POSTPROCESSING = 'postprocessing'
    LOSSES = 'losses'
    SUMMARY = 'summary'
    METRIC = 'metric'
    TRAIN_OP = 'train_op'
    GRADIENTS = 'gradients'
    MODEL = 'model'
    PREDICTIONS_RAW = 'predictions_raw'
    DATASET = 'dataset'


# pylint: disable=too-few-public-methods
# it is only a container, so no methods required
class CollectionNames:
    """
    Names of the model scopes
    """
    INPUTS = 'nc7_inputs'
    INPUTS_PREPROCESSED = 'nc7_inputs_preprocessed'
    PREDICTIONS = 'nc7_predictions'
    PREDICTIONS_RAW = 'nc7_predictions_raw'
    LOSSES = 'nc7_losses'
    SUMMARY = 'nc7_summary'
    METRIC = 'nc7_metric'
