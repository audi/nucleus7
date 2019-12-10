# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Utils to work with numpy arrays
"""

from typing import List
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np


def dense_to_sparse(array: np.ndarray, sparse_value_as_zeroes=0
                    ) -> Tuple[np.ndarray, np.ndarray, tuple]:
    """
    take dense array a and returns dict with indices, values and shape

    Parameters
    ----------
    array
        dense array
    sparse_value_as_zeroes
        value to use as a zero inside of generated sparse tensor

    Returns
    -------
    sparse_indices
        indices for sparse tensor
    sparse_values
        values for sparse tensor
    sparse_shape
        shape of the sparse tensor
    """
    # pylint: disable=singleton-comparison
    # this if format for numpy comparison
    indices = np.where(np.isclose(array, sparse_value_as_zeroes) == False)
    if indices[0].shape[0] == 0:
        indices = (0, 0, 0)
    values = array[indices]
    shape = array.shape
    return np.transpose(indices), values, shape


def fig2rgb_array(fig: plt.Figure, expand_batch_dimension: bool = True
                  ) -> np.ndarray:
    """
    Convert matplotlib figure to numpy RGB array

    Parameters
    ----------
    fig
        figure to convert
    expand_batch_dimension
        if the single batch dimension should be added

    Returns
    -------
    figure_as_array
        figure as numpy array

    """
    fig.canvas.draw()
    buf = fig.canvas.tostring_rgb()
    ncols, nrows = fig.canvas.get_width_height()
    shape = ((nrows, ncols, 3) if not expand_batch_dimension
             else (1, nrows, ncols, 3))
    return np.fromstring(buf, dtype=np.uint8).reshape(shape)


def stack_with_pad(arrays: List[np.ndarray], axis=0, pad_value=0) -> np.ndarray:
    """
    Stack arrays and if they have equal number of dimensions, but different
    shapes, pad them with pad_value to the maximum shape

    Parameters
    ----------
    arrays
        list of arrays to stack
    axis
        axis to stack
    pad_value
        value to use if arrays must be padded

    Returns
    -------
    stacked_arrays
        stacked arrays

    Raises
    ------
    ValueError
        if arrays is empty
    ValueError
        if arrays have different number of dimensions
    """
    if not arrays:
        raise ValueError("Provide at least one array to stack!")
    arrays = [np.asarray(each_array) for each_array in arrays]
    if len({each_array.ndim for each_array in arrays}) > 1:
        raise ValueError("Only arrays with same number of dimensions can be"
                         "stacked and padded!")

    try:
        arrays_stacked = np.stack(arrays, axis=axis)
    except ValueError:
        ndim = arrays[0].ndim
        max_shape = np.array(
            [max([each_array.shape[i_dim] for each_array in arrays])
             for i_dim in range(ndim)])
        paddings = [[[0, max_shape[i_dim] - each_array.shape[i_dim]]
                     for i_dim in range(ndim)]
                    for each_array in arrays]
        arrays_padded = [np.pad(each_array, each_padding, "constant",
                                constant_values=pad_value)
                         for each_array, each_padding in zip(arrays, paddings)]
        arrays_stacked = np.stack(arrays_padded, axis)
    return arrays_stacked
