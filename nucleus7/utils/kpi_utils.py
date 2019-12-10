# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Utils for KPI calculation
"""

import logging
import numbers

import numpy as np

from nucleus7.utils import nest_utils


def filter_kpi_values(kpi: dict,
                      return_flattened: bool = False) -> (dict, dict):
    """
    Filter kpis according to its value type. If kpi value is not of type str
    or number, it will be filtered out. If value is numpy array of size 1, then
    element will be selected and not filtered out.

    Parameters
    ----------
    kpi
        dict, possibly nested, mapping kpi names to its values
    return_flattened
        flag to return flattened dict and do not unflatten it back

    Returns
    -------
    kpi_filtered
        dict with same structure as kpi, but only with values of numbers and
        string type
    kpi_filtered_out
        dict with same structure as kpi with values other then numbers and
        string type
    """
    logger = logging.getLogger(__name__)
    kpi_flatten = nest_utils.flatten_nested_struct(kpi)

    names_filtered_out = []
    for kpi_name in kpi_flatten:
        kpi_value = kpi_flatten[kpi_name]
        if isinstance(kpi_value, np.ndarray) and np.prod(kpi_value.shape) == 1:
            kpi_value = np.reshape(kpi_value, (1,))[0]
        # pylint: disable=no-member
        # numpy does have floating member
        if isinstance(kpi_value, np.floating):
            kpi_value = float(kpi_value)
        elif isinstance(kpi_value, np.integer):
            kpi_value = int(kpi_value)
        elif isinstance(kpi_value, np.str):
            kpi_value = str(kpi_value)
        kpi_flatten[kpi_name] = kpi_value
        if not isinstance(kpi_value, (numbers.Number, str, list)):
            names_filtered_out.append(kpi_name)
    kpi_filtered = {k: v for k, v in kpi_flatten.items()
                    if k not in names_filtered_out}
    kpi_filtered_out = {k: v for k, v in kpi_flatten.items()
                        if k in names_filtered_out}
    if kpi_filtered and not return_flattened:
        kpi_filtered = nest_utils.unflatten_dict_to_nested(kpi_filtered)
    if kpi_filtered_out:
        logger.warning("Following kpi keys cannot be serialized to json: "
                       "%s", kpi_filtered_out.keys())
        if not return_flattened:
            kpi_filtered_out = nest_utils.unflatten_dict_to_nested(
                kpi_filtered_out)
    return kpi_filtered, kpi_filtered_out
