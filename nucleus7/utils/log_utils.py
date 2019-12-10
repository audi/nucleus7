# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Utils for logging
"""
from functools import wraps
import logging
from typing import Callable
from typing import Optional

import numpy as np
import tensorflow as tf

from nucleus7.utils import nest_utils


def log_nucleotide_inputs_outputs(log_level: str = "debug",
                                  shape_fn: Callable = np.shape,
                                  logger: Optional[logging.Logger] = None
                                  ) -> Callable:
    """
    Log nucleotide inputs and outputs

    Parameters
    ----------
    log_level
        level of the logging to use; should be one of [info, debug]
    shape_fn
        function to get the shape for objects other than tensorflow tensors
    logger
        logger to use

    Returns
    -------
    wrapped_function
        wrapped function
    """
    logger = logger or logging.getLogger(__name__)
    logger_fn = {
        "info": logger.info,
        "debug": logger.debug
    }[log_level.lower()]

    def wrapped(function: Callable):
        @wraps(function)
        def wrapper(self, *args, **inputs):
            function_name = function.__name__
            logger_fn("%s: %s in mode %s received kwargs inputs: %s",
                      self.name, function_name, self.mode,
                      _format_data_for_log(inputs, shape_fn))
            result = function(self, *args, **inputs)
            if result is None:
                logger_fn("%s: %s in mode %s has no results",
                          self.name, function_name, self.mode)
            else:
                logger_fn("%s: %s in mode %s generated outputs: %s",
                          self.name, function_name, self.mode,
                          _format_data_for_log(result, shape_fn))
            return result

        return wrapper

    return wrapped


def _format_data_for_log(data: Optional[dict] = None,
                         shape_fn: Callable = np.shape
                         ) -> Optional[dict]:
    if data is None:
        return None
    data_flat = nest_utils.flatten_nested_struct(data)
    data_repr_flat = {k: (v if isinstance(v, tf.Tensor) else shape_fn(v))
                      for k, v in data_flat.items()}
    data_repr = nest_utils.unflatten_dict_to_nested(data_repr_flat)
    return data_repr
